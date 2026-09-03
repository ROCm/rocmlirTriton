//===- LLMProposer.cpp - Running the config-proposing helper --------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLMProposer.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <limits>
#include <thread>

#ifndef _WIN32
#include <csignal>
#endif

#define DEBUG_TYPE "rock-llm-search"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Where the helper lives, relative to the directory holding this binary:
/// `ci-performance-scripts` copies the package to ${ROCMLIR_BIN_DIR}/llm,
/// beside the other performance scripts, and both the build and install trees
/// have it there. The source tree is deliberately not searched -- a driver
/// running against a stale checkout of the helper is worse than one that says
/// it cannot find it, and `kProposerEnvVar` covers pointing it elsewhere on
/// purpose.
constexpr StringLiteral kProposerRelPath = "llm/proposer.py";

/// Overrides the helper's location. This is what lets a test drive the search
/// with a script that answers from a fixture instead of from a model, which is
/// the only way any of this runs offline.
constexpr StringLiteral kProposerEnvVar = "ROCMLIR_LLM_PROPOSER";

/// How often the wait loop wakes to check whether the helper has finished.
/// Short enough that a fast reply is not sat on, long enough that waiting for
/// a slow model costs nothing.
constexpr unsigned kPollIntervalMs = 50;

/// A proposal reply contains only a small JSON object and a batch of short
/// perf-config strings. Bound it before parsing so a broken helper cannot make
/// the compiler allocate or recursively parse an unbounded response.
constexpr uint64_t kMaxResponseBytes = 1024 * 1024;
constexpr unsigned kMaxJsonNesting = 64;
constexpr size_t kMaxPerfConfigBytes = 4096;

std::string thisExecutableDir() {
  std::string exePath = llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&thisExecutableDir));
  return std::string(llvm::sys::path::parent_path(exePath));
}

/// Whatever the helper had to say about itself, for an error message. Empty
/// when it said nothing, which is the usual case for a timeout.
std::string diagnosticsOf(StringRef logPath) {
  auto buffer = llvm::MemoryBuffer::getFile(logPath);
  if (!buffer)
    return {};
  return (*buffer)->getBuffer().trim().str();
}

/// Runs `args` with its output going to `logPath`, and kills it if it outlives
/// `timeoutSec`.
///
/// Follows `compileConfigViaSubprocess` in rocmlir-tuning-driver rather than
/// `sys::ExecuteAndWait`'s own timeout, whose SIGALRM is process-global; see
/// the reasoning there. A search runs one helper at a time, so the hazard is
/// smaller here, but the idiom is the same and there is no reason to keep two.
llvm::Error runWithTimeout(ArrayRef<StringRef> args, StringRef logPath,
                           unsigned timeoutSec) {
  StringRef program = args[0];
  auto failed = [&](const llvm::Twine &what) {
    std::string diagnostics = diagnosticsOf(logPath);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "%s",
        (program + " " + what + (diagnostics.empty() ? "" : "\n") + diagnostics)
            .str()
            .c_str());
  };

  // stdin from /dev/null so a helper that reads it cannot wait on a terminal;
  // stdout and stderr to one file, since both are only ever read to explain a
  // failure and interleaving them is how they were written.
  std::optional<StringRef> redirects[3] = {StringRef(""), logPath, logPath};
  std::string execErr;
  bool execFailed = false;
  llvm::sys::ProcessInfo proc =
      llvm::sys::ExecuteNoWait(program, args, /*Env=*/std::nullopt, redirects,
                               /*MemoryLimit=*/0, &execErr, &execFailed);
  if (execFailed || proc.Pid == llvm::sys::ProcessInfo::InvalidPid)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot run %s: %s", program.str().c_str(),
                                   execErr.c_str());

  // Kills the helper and reaps it, so that a timeout does not leave a process
  // talking to a model nobody is listening to.
  auto killAndReap = [&] {
#ifdef _WIN32
    // A finite wait terminates a child still running when it expires.
    const std::optional<unsigned> secondsToWait = 1;
#else
    kill(proc.Pid, SIGKILL);
    const std::optional<unsigned> secondsToWait = std::nullopt;
#endif
    llvm::sys::Wait(proc, secondsToWait, /*ErrMsg=*/nullptr);
  };

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  while (true) {
    std::string waitErr;
    // SecondsToWait=0 is a non-blocking poll: Pid is 0 while the child runs
    // and its own once it exits. Polling=true keeps a transient EINTR from
    // making LLVM kill it.
    llvm::sys::ProcessInfo done =
        llvm::sys::Wait(proc, /*SecondsToWait=*/0, &waitErr,
                        /*ProcStat=*/nullptr, /*Polling=*/true);
    if (done.Pid == proc.Pid) {
      if (done.ReturnCode == 0)
        return llvm::Error::success();
      return failed("failed with exit code " + llvm::Twine(done.ReturnCode) +
                    ":");
    }
    if (done.Pid < 0 && waitErr.empty()) {
      // On POSIX LLVM returns Pid=-1 and leaves the error empty when wait4's
      // non-blocking poll is interrupted. The child is still healthy.
    } else if (done.Pid != 0) {
      // Neither "still running" nor "our child": a genuine wait failure.
      killAndReap();
      return failed("could not be waited for (" + waitErr + "):");
    }
    if (timeoutSec > 0 && std::chrono::steady_clock::now() >= deadline) {
      killAndReap();
      return failed("did not answer within " + llvm::Twine(timeoutSec) + "s:");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }
}

/// Reads the `configs` array of a helper's reply. Everything the model itself
/// got wrong has already been dealt with on the Python side; what is left to
/// check here is that the helper kept its own end of the contract.
llvm::Expected<std::vector<std::string>> parseResponse(StringRef text,
                                                       unsigned maxConfigs,
                                                       StringRef program,
                                                       StringRef responsePath) {
  auto invalid = [&](const llvm::Twine &what) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "%s wrote an unusable reply to %s: %s",
        program.str().c_str(), responsePath.str().c_str(), what.str().c_str());
  };
  unsigned nesting = 0;
  bool inString = false;
  bool escaped = false;
  for (char c : text) {
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
    } else if (c == '[' || c == '{') {
      if (++nesting > kMaxJsonNesting)
        return invalid("reply is nested too deeply");
    } else if ((c == ']' || c == '}') && nesting > 0) {
      --nesting;
    }
  }

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed)
    return invalid(llvm::toString(parsed.takeError()));

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("expected a JSON object");
  // A failure the helper can explain -- no API key, the model run failed --
  // comes back in-band, so that it can be reported in the helper's own words
  // rather than as an exit code and a log to go and read. Note that a model
  // which answered with nothing *usable* is not this: that is an empty
  // `configs` array, and an empty round is a search result, not a bug.
  if (const llvm::json::Value *error = root->get("error")) {
    std::optional<StringRef> message = error->getAsString();
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "%s",
        message.value_or("unspecified").str().c_str());
  }
  const llvm::json::Array *configs = root->getArray("configs");
  if (!configs)
    return invalid("reply has no 'configs' array");
  if (configs->size() > maxConfigs)
    return invalid("reply has " + llvm::Twine(configs->size()) +
                   " configs, but only " + llvm::Twine(maxConfigs) +
                   " were requested");

  // Whole configs in the canonical named form, which is the only spelling
  // anything in-tree accepts. Whether one of them names a legal config is not
  // asked here: that is the tuning space's question, and `LLMSearch` puts it
  // to `TuningParamAxes::isFeasible`. All that is checked is that the helper
  // answered with strings, since a helper that did not is broken rather than
  // unlucky.
  std::vector<std::string> result;
  for (const llvm::json::Value &entry : *configs) {
    std::optional<StringRef> perfConfig = entry.getAsString();
    if (!perfConfig)
      return invalid(
          "a config is not a string; the helper answers with perf configs "
          "such as 'gemm:mPerBlock=128,...'");
    if (perfConfig->size() > kMaxPerfConfigBytes)
      return invalid("a config is " + llvm::Twine(perfConfig->size()) +
                     " bytes; the maximum is " +
                     llvm::Twine(kMaxPerfConfigBytes));
    result.push_back(perfConfig->str());
  }
  return result;
}

} // namespace

std::string LLMProposer::resolveProgram(StringRef program) {
  if (!program.empty())
    return program.str();
  if (const char *fromEnv = std::getenv(kProposerEnvVar.data()))
    return std::string(fromEnv);

  // Returned whether or not it is there: when it is not, this is the path the
  // "cannot run" error should name, so that it points at where the helper was
  // expected rather than at nothing.
  SmallString<128> path(thisExecutableDir());
  llvm::sys::path::append(path, kProposerRelPath);
  return std::string(path);
}

LLMProposer::LLMProposer(StringRef program, StringRef sessionPath,
                         StringRef transcriptPath, unsigned timeoutSec)
    : program(resolveProgram(program)), sessionPath(sessionPath.str()),
      transcriptPath(transcriptPath.str()), timeoutSec(timeoutSec) {}

llvm::Expected<std::vector<std::string>>
LLMProposer::propose(llvm::json::Object request) {
  std::optional<int64_t> requested = request.getInteger("configsRequested");
  if (!requested || *requested < 0 ||
      static_cast<uint64_t>(*requested) > std::numeric_limits<unsigned>::max())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "the request has no valid unsigned 'configsRequested'");
  unsigned maxConfigs = static_cast<unsigned>(*requested);

  SmallString<128> requestPath, responsePath, logPath;
  if (llvm::sys::fs::createTemporaryFile("rocmlir-llm-req", "json",
                                         requestPath) ||
      llvm::sys::fs::createTemporaryFile("rocmlir-llm-resp", "json",
                                         responsePath) ||
      llvm::sys::fs::createTemporaryFile("rocmlir-llm-log", "log", logPath))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot create the helper's temp files");
  llvm::FileRemover removeRequest(requestPath);
  llvm::FileRemover removeResponse(responsePath);
  llvm::FileRemover removeLog(logPath);

  {
    std::error_code ec;
    llvm::raw_fd_ostream out(requestPath, ec, llvm::sys::fs::OF_Text);
    if (ec)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot write %s: %s", requestPath.c_str(),
                                     ec.message().c_str());
    out << llvm::json::Value(std::move(request));
  }

  std::string requestArg = ("--request=" + requestPath).str();
  std::string responseArg = ("--response=" + responsePath).str();
  std::string sessionArg = ("--session=" + sessionPath);
  std::string transcriptArg = ("--transcript=" + transcriptPath);
  SmallVector<StringRef, 6> args = {program, requestArg, responseArg};
  if (!sessionPath.empty())
    args.push_back(sessionArg);
  if (!transcriptPath.empty())
    args.push_back(transcriptArg);

  LLVM_DEBUG(llvm::dbgs() << "LLM: running " << program << " on " << requestPath
                          << "\n");
  if (llvm::Error err = runWithTimeout(args, logPath, timeoutSec))
    return std::move(err);

  uint64_t responseSize = 0;
  if (std::error_code ec = llvm::sys::fs::file_size(responsePath, responseSize))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot inspect %s: %s",
                                   responsePath.c_str(), ec.message().c_str());
  if (responseSize > kMaxResponseBytes)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "%s wrote an oversized reply to %s (%llu bytes; maximum %llu)",
        program.c_str(), responsePath.c_str(),
        static_cast<unsigned long long>(responseSize),
        static_cast<unsigned long long>(kMaxResponseBytes));

  auto buffer = llvm::MemoryBuffer::getFile(responsePath);
  if (!buffer)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s wrote no reply to %s", program.c_str(),
                                   responsePath.c_str());
  return parseResponse((*buffer)->getBuffer(), maxConfigs, program,
                       responsePath);
}

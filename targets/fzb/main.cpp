#include "cli/progress_line.h"
#include "compile/exact_eval.h"
#include "compile/ftp/compile.h"
#include "compile/nombrilo.h"
#include "compile/pcfb_writer.h"
#include "compile/sha256.h"
#include "runtime/progress_log.h"
#include "version.h"

#ifdef FZB_HAS_RUNTIME
#include "runtime/runtime.h"
#endif

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace po = boost::program_options;

namespace {

namespace progress = fuzzybits::runtime::progress;

/// `--no-progress` / `--quiet`, shared by every command that can show progress.
struct progress_flags {
  bool no_progress = false;
  bool quiet = false;

  /// Attach a renderer to `sinks` when the policy allows it. Returns nullptr (and leaves `sinks`
  /// alone) for pipes, `--no-progress`, `--quiet` and `FUZZYBITS_PROGRESS=0`.
  [[nodiscard]] std::unique_ptr<progress::line_sink> attach(std::vector<progress::sink *> &sinks) const {
    if (!progress::resolve_display(no_progress, quiet).enabled()) {
      return nullptr;
    }
    auto sink = std::make_unique<progress::line_sink>(2);
    sinks.push_back(sink.get());
    return sink;
  }

  /// Informational stderr (`wrote …`, the run summary) is dropped by `--quiet`; results and
  /// errors still print.
  [[nodiscard]] bool silent() const { return quiet; }
};

void add_progress_options(po::options_description &desc) {
  desc.add_options()
      ("no-progress", "do not draw the progress line (it is on by default on a terminal)")
      ("quiet,q", "no progress line and no informational stderr; results and errors still print");
}

progress_flags read_progress_flags(const po::variables_map &vm) {
  progress_flags flags;
  flags.no_progress = vm.count("no-progress") > 0;
  flags.quiet = vm.count("quiet") > 0;
  return flags;
}

void print_root_usage(std::ostream &out) {
  out << "Usage: fzb <command> [options]\n\n"
      << "Commands:\n"
      << "  compile   OpenPSA/MEF XML, nombrilo.circuit or CAFTA FTP projects → .pcfb\n"
      << "  export    .pcfb → nombrilo.circuit (.safetensors)\n"
      << "  inspect   Summarise, validate, or exactly evaluate a .pcfb\n"
      << "  run       Monte Carlo a .pcfb (estimate and CI per query)\n";
}

std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          std::ostringstream os;
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
          out += os.str();
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

constexpr const char *kProgressLogHelp =
    "write a progress log to FILE (.jsonl or .csv; all snapshot fields, "
    "independent of the terminal display)";

/// Opens the opt-in `--progress-log FILE`. Returns null when the flag is absent,
/// so no file is ever created unless asked for. Throws `std::runtime_error` /
/// `std::invalid_argument` (bad extension) before any work starts.
std::unique_ptr<fuzzybits::runtime::progress::log_sink> open_progress_log(const po::variables_map &vm) {
  if (vm.count("progress-log") == 0) {
    return nullptr;
  }
  return std::make_unique<fuzzybits::runtime::progress::log_sink>(
      std::filesystem::path(vm["progress-log"].as<std::string>()));
}

#ifdef FZB_HAS_RUNTIME
std::uint64_t parse_trials(const std::string &text) {
  char *end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || v <= 0 || !std::isfinite(v)) {
    throw fuzzybits::runtime::RuntimeError("invalid --trials '" + text + "'");
  }
  return static_cast<std::uint64_t>(v);
}

int cmd_run(int argc, char **argv) {
  po::options_description desc("fzb run options");
  desc.add_options()
      ("help,h", "display this help message")
      ("trials,N", po::value<std::string>(), "Bernoulli trials (e.g. 1e9, 1000000)")
      ("seed,S", po::value<int>()->default_value(372),
       "must be 372 (the scram-cli default); other values are rejected")
      ("backend", po::value<std::string>(), "AdaptiveCpp backend: cuda, hip, metal or omp (sets ACPP_VISIBILITY_MASK)")
      ("json", "print results as JSON")
      ("progress-log", po::value<std::string>(), kProgressLogHelp)
      ("input-file", po::value<std::string>(), ".pcfb file");
  add_progress_options(desc);

  po::positional_options_description pos;
  pos.add("input-file", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb run model.pcfb --trials N [--seed S] [--backend cuda|hip|metal|omp] [--json]\n"
                 "              [--no-progress] [--quiet]\n\n"
              << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb run: missing .pcfb file\n\n" << desc << std::endl;
    return 1;
  }
  if (!vm.count("trials")) {
    std::cerr << "fzb run: --trials is required\n\n" << desc << std::endl;
    return 1;
  }

  fuzzybits::runtime::RunOptions opt;
  try {
    opt.trials = parse_trials(vm["trials"].as<std::string>());
  } catch (const std::exception &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }
  opt.seed = vm["seed"].as<int>();
  if (opt.seed != 372) {
    std::cerr << "fzb run: --seed is not supported yet; results are deterministic "
                 "per model and trial count (Philox counters are "
                 "{pdag_idx+1, event_idx+1, sample_idx+1, (iteration+1)<<6}, "
                 "not Settings::seed). The only accepted value is 372, the "
                 "scram-cli default.\n";
    return 1;
  }
  if (vm.count("backend")) {
    opt.backend = vm["backend"].as<std::string>();
  }

  // ACPP_VISIBILITY_MASK must be set before AdaptiveCpp constructs a queue.
  try {
    fuzzybits::runtime::apply_backend_mask(opt.backend);
  } catch (const std::exception &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }

  // The log is opt-in and independent of the terminal display (`--no-progress`,
  // `--quiet` and redirection do not affect it). Opening it here fails before
  // sampling and before the progress line draws anything.
  std::unique_ptr<fuzzybits::runtime::progress::log_sink> progress_log;
  try {
    progress_log = open_progress_log(vm);
  } catch (const std::exception &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }
  const progress_flags pf = read_progress_flags(vm);
  auto progress_sink = pf.attach(opt.progress_sinks);
  if (progress_log) {
    opt.progress_sinks.push_back(progress_log.get());
  }

  try {
    const auto results = fuzzybits::runtime::run_pcfb(vm["input-file"].as<std::string>(), opt);
    if (progress_log) {
      // No-op when the engine already delivered its finish record.
      fuzzybits::runtime::progress::finish_log(*progress_log, results);
    }
    if (progress_log && !progress_log->ok()) {
      std::cerr << "fzb run: writing the progress log failed: " << progress_log->path().string()
                << std::endl;
      return 1;
    }
    if (vm.count("json")) {
      std::cout << "{\n  \"queries\": [\n";
      for (std::size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        std::cout << "    {\"name\": \"" << json_escape(r.name) << "\", "
                  << "\"estimate\": " << std::setprecision(16) << std::scientific << r.estimate << ", "
                  << "\"ci_low\": " << r.ci_low << ", "
                  << "\"ci_high\": " << r.ci_high << ", "
                  << "\"n\": " << std::dec << r.n << ", "
                  << "\"hits\": " << r.hits << "}";
        if (i + 1 < results.size()) {
          std::cout << ",";
        }
        std::cout << "\n";
      }
      std::cout << "  ]\n}\n";
    } else {
      for (const auto &r : results) {
        std::cout << r.name << "  " << std::setprecision(8) << std::scientific << r.estimate << "  ["
                  << r.ci_low << ", " << r.ci_high << "]  n=" << std::dec << r.n << "  hits=" << r.hits
                  << "\n";
      }
    }
    return 0;
  } catch (const fuzzybits::runtime::InvalidPcfbError &ex) {
    if (progress_sink) {
      progress_sink->stop();  // erase the block before the error text lands in it
    }
    if (progress_log) {
      fuzzybits::runtime::progress::finish_log(*progress_log, /*interrupted=*/true);
    }
    std::cerr << "fzb run: " << ex.what() << std::endl;
    for (const auto &d : ex.diagnostics) {
      std::cerr << "validate: " << d.location << ": " << d.message << std::endl;
    }
    return 1;
  } catch (const std::exception &ex) {
    if (progress_sink) {
      progress_sink->stop();
    }
    if (progress_log) {
      // A failed run still closes its log, marked interrupted, instead of
      // leaving a file that looks like a run in progress.
      fuzzybits::runtime::progress::finish_log(*progress_log, /*interrupted=*/true);
    }
    // A partial run has already printed its own summary through on_finish; `--quiet` silences
    // only informational lines, never an error.
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }
}
#else
int cmd_run(int, char **) {
  std::cerr << "fzb run requires FUZZYBITS_WITH_MC=ON (the SYCL Monte Carlo engine).\n";
  return 2;
}
#endif

const char *opcode_name(pcfb::OpCode op) {
  switch (op) {
    case pcfb::OpCode::BITWISE_NO_OP:
      return "NO_OP";
    case pcfb::OpCode::BITWISE_NOT:
      return "NOT";
    case pcfb::OpCode::BITWISE_AND:
      return "AND";
    case pcfb::OpCode::BITWISE_OR:
      return "OR";
    case pcfb::OpCode::BITWISE_XOR:
      return "XOR";
    case pcfb::OpCode::BITWISE_NAND:
      return "NAND";
    case pcfb::OpCode::BITWISE_NOR:
      return "NOR";
    case pcfb::OpCode::BITWISE_XNOR:
      return "XNOR";
    case pcfb::OpCode::BITWISE_ATLEAST:
      return "ATLEAST";
    case pcfb::OpCode::BITWISE_ATMOST:
      return "ATMOST";
    case pcfb::OpCode::BITWISE_EXACTLY:
      return "EXACTLY";
    case pcfb::OpCode::BITWISE_ZEROS:
      return "ZEROS";
    case pcfb::OpCode::BITWISE_ONES:
      return "ONES";
    case pcfb::OpCode::SAMPLE:
      return "SAMPLE";
    default:
      return "RESERVED";
  }
}

const char *source_name(pcfb::SampleSource s) {
  switch (s) {
    case pcfb::SampleSource::BASIC_EVENT:
      return "BASIC_EVENT";
    case pcfb::SampleSource::COLLAPSED_MODULE:
      return "COLLAPSED_MODULE";
    case pcfb::SampleSource::HOUSE_EVENT:
      return "HOUSE_EVENT";
    default:
      return "UNKNOWN";
  }
}

struct InspectSummary {
  std::uint32_t schema_version = 0;
  std::size_t n_tensors = 0;
  std::size_t n_ops = 0;
  std::size_t n_queries = 0;
  std::size_t n_samples = 0;
  std::map<std::string, int> opcode_counts;
  std::map<std::string, int> sample_sources;
  std::map<std::string, std::string> metadata;
  std::vector<std::string> queries;
};

InspectSummary summarise(const pcfb::ir::Circuit &circuit) {
  InspectSummary s;
  s.schema_version = circuit.schema_version();
  if (circuit.get() && circuit.get()->metadata()) {
    for (const auto *kv : *circuit.get()->metadata()) {
      if (!kv || !kv->key()) {
        continue;
      }
      std::string value;
      if (const auto *str = kv->value_as_StringValue(); str && str->value()) {
        value = str->value()->str();
      } else if (const auto *i = kv->value_as_IntValue()) {
        value = std::to_string(i->value());
      } else if (const auto *f = kv->value_as_FloatValue()) {
        std::ostringstream os;
        os << f->value();
        value = os.str();
      } else if (const auto *b = kv->value_as_BytesValue(); b && b->value()) {
        std::vector<std::uint8_t> raw(b->value()->begin(), b->value()->end());
        if (raw.size() == 32) {
          std::array<std::uint8_t, 32> digest{};
          std::copy(raw.begin(), raw.end(), digest.begin());
          value = fuzzybits::compile::sha256_hex(digest);
        } else {
          value = "<" + std::to_string(raw.size()) + " bytes>";
        }
      }
      s.metadata.emplace(kv->key()->str(), std::move(value));
    }
  }
  if (!circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
    return s;
  }
  const auto *sc = circuit.subcircuits()->Get(0);
  if (sc->tensors()) {
    s.n_tensors = sc->tensors()->size();
  }
  if (sc->operators()) {
    s.n_ops = sc->operators()->size();
    for (const auto *op : *sc->operators()) {
      if (!op) {
        continue;
      }
      ++s.opcode_counts[opcode_name(op->opcode())];
      if (op->opcode() == pcfb::OpCode::SAMPLE) {
        ++s.n_samples;
        if (const auto *sa = op->args_as_SampleArgs()) {
          ++s.sample_sources[source_name(sa->source())];
        }
      }
    }
  }
  if (sc->queries()) {
    s.n_queries = sc->queries()->size();
    for (const auto *q : *sc->queries()) {
      if (!q) {
        continue;
      }
      std::string name = q->name() ? q->name()->str() : "";
      s.queries.push_back(name.empty() ? ("t" + std::to_string(q->target())) : name);
    }
  }
  return s;
}

void print_text(const InspectSummary &s, std::ostream &out) {
  out << "schema_version  0x" << std::hex << s.schema_version << std::dec << "\n";
  out << "tensors         " << s.n_tensors << "\n";
  out << "operators       " << s.n_ops << "\n";
  out << "samples         " << s.n_samples << "\n";
  out << "queries         " << s.n_queries << "\n";
  if (!s.opcode_counts.empty()) {
    out << "opcodes\n";
    for (const auto &[name, n] : s.opcode_counts) {
      out << "  " << name << "  " << n << "\n";
    }
  }
  if (!s.sample_sources.empty()) {
    out << "sample_source\n";
    for (const auto &[name, n] : s.sample_sources) {
      out << "  " << name << "  " << n << "\n";
    }
  }
  if (!s.queries.empty()) {
    out << "query_names\n";
    for (const auto &q : s.queries) {
      out << "  " << q << "\n";
    }
  }
  if (!s.metadata.empty()) {
    out << "metadata\n";
    for (const auto &[k, v] : s.metadata) {
      out << "  " << k << "  " << v << "\n";
    }
  }
}

void print_json(const InspectSummary &s, std::ostream &out) {
  out << "{\n";
  out << "  \"schema_version\": " << s.schema_version << ",\n";
  out << "  \"tensors\": " << s.n_tensors << ",\n";
  out << "  \"operators\": " << s.n_ops << ",\n";
  out << "  \"samples\": " << s.n_samples << ",\n";
  out << "  \"queries\": " << s.n_queries << ",\n";
  out << "  \"opcodes\": {";
  bool first = true;
  for (const auto &[name, n] : s.opcode_counts) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(name) << "\": " << n;
  }
  out << "},\n  \"sample_source\": {";
  first = true;
  for (const auto &[name, n] : s.sample_sources) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(name) << "\": " << n;
  }
  out << "},\n  \"query_names\": [";
  first = true;
  for (const auto &q : s.queries) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(q) << "\"";
  }
  out << "],\n  \"metadata\": {";
  first = true;
  for (const auto &[k, v] : s.metadata) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(k) << "\": \"" << json_escape(v) << "\"";
  }
  out << "}\n}\n";
}

std::string read_text_file(const std::filesystem::path &path, std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot read " + path.string();
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/// The FTP project path of `fzb compile`: reader, quantification config, lowering, passes, writer.
int compile_ftp_inputs(const po::variables_map &vm, const std::vector<std::string> &files,
                       const progress_flags &pf,
                       const std::unique_ptr<fuzzybits::runtime::progress::log_sink> &progress_log) {
  namespace ftp = fuzzybits::compile::ftp;

  const auto fail = [](const std::string &message) {
    std::cerr << "fzb compile: " << message << std::endl;
    return 1;
  };
  for (const char *flag : {"no-kn", "no-xor", "recognize-atleast", "no-recognize-atleast", "recognize-xor",
                           "no-recognize-xor", "keep-null", "ccf", "allow-extern", "weights"}) {
    if (vm.count(flag)) {
      return fail(std::string("--") + flag + " applies to OpenPSA/MEF input, not to FTP projects");
    }
  }

  ftp::options opt;
  if (vm.count("target")) {
    opt.targets = vm["target"].as<std::vector<std::string>>();
  }
  if (vm.count("watch")) {
    opt.watch = vm["watch"].as<std::vector<std::string>>();
  }
  if (vm.count("override")) {
    for (const auto &text : vm["override"].as<std::vector<std::string>>()) {
      auto parsed = ftp::parse_override(text);
      if (!parsed) {
        return fail(parsed.diagnostics().items().front().message);
      }
      opt.overrides.push_back(std::move(parsed).value());
    }
  }
  if (vm.count("duplicate-import")) {
    opt.duplicate_import = vm["duplicate-import"].as<std::string>();
  }
  if (vm.count("complement-events")) {
    opt.complement_events = vm["complement-events"].as<std::string>();
  }
  if (vm.count("negation")) {
    opt.negation = vm["negation"].as<std::string>();
  }
  if (vm.count("literals")) {
    opt.literals = vm["literals"].as<std::string>();
  }
  if (vm.count("max-initiators-per-cutset")) {
    const auto text = vm["max-initiators-per-cutset"].as<std::string>();
    if (text == "none") {
      opt.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    } else {
      std::uint32_t count = 0;
      const auto *end = text.data() + text.size();
      const auto [pointer, status] = std::from_chars(text.data(), end, count);
      if (status != std::errc{} || pointer != end || text.empty()) {
        return fail("--max-initiators-per-cutset must be a non-negative integer or none, not '" + text + "'");
      }
      opt.max_initiators_per_cutset = std::optional<std::uint32_t>{count};
    }
  }
  if (!vm["compilation-passes"].defaulted()) {
    opt.level = vm["compilation-passes"].as<int>();
  }
  if (vm.count("config")) {
    std::string error;
    auto text = read_text_file(vm["config"].as<std::string>(), error);
    if (!error.empty()) {
      return fail(error);
    }
    opt.config = std::move(text);
  }

  auto progress_sink = pf.attach(opt.progress_sinks);
  if (progress_log) {
    opt.progress_sinks.push_back(progress_log.get());
  }
  std::vector<std::filesystem::path> paths(files.begin(), files.end());
  auto compiled = ftp::compile_project(paths, opt);

  const auto write_report = [&]() {
    if (!vm.count("report")) {
      return true;
    }
    std::ofstream report(vm["report"].as<std::string>(), std::ios::binary);
    report << ftp::report_json(compiled) << "\n";
    report.flush();
    return static_cast<bool>(report);
  };

  std::size_t warnings = 0;
  std::size_t notes = 0;
  constexpr std::size_t kMaxPrintedWarnings = 10;
  // Errors always print; warnings only without --quiet, the first few.
  const auto print_diagnostics = [&]() {
    std::size_t printed = 0;
    for (const auto &item : compiled.diagnostics().items()) {
      switch (item.severity) {
        case fuzzybits::api::diagnostic_severity::error:
          std::cerr << "error: " << item.code << ": " << item.message << "\n";
          break;
        case fuzzybits::api::diagnostic_severity::warning:
          ++warnings;
          if (!pf.silent() && printed++ < kMaxPrintedWarnings) {
            std::cerr << "warning: " << item.code << ": " << item.message << "\n";
          }
          break;
        case fuzzybits::api::diagnostic_severity::note:
          ++notes;
          break;
      }
    }
    if (!pf.silent() && warnings > kMaxPrintedWarnings) {
      std::cerr << "... " << (warnings - kMaxPrintedWarnings) << " more warning(s); --report FILE lists all\n";
    }
  };
  if (!compiled) {
    if (progress_sink) {
      progress_sink->stop();
    }
    print_diagnostics();
    if (!write_report()) {
      std::cerr << "fzb compile: writing the report failed: " << vm["report"].as<std::string>() << std::endl;
    }
    return fail("FTP compile failed");
  }

  const auto &result = compiled.value();
  std::filesystem::path out_path;
  if (vm.count("output")) {
    out_path = vm["output"].as<std::string>();
  } else {
    out_path = std::filesystem::path(result.model_name).filename();
    out_path += ".pcfb";
  }
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    if (progress_sink) {
      progress_sink->stop();
    }
    return fail("cannot write " + out_path.string());
  }
  out.write(reinterpret_cast<const char *>(result.bytes.data()),
            static_cast<std::streamsize>(result.bytes.size()));
  out.flush();
  if (!out) {
    if (progress_sink) {
      progress_sink->stop();
    }
    return fail("write failed: " + out_path.string());
  }
  if (progress_sink) {
    progress_sink->finish_now(/*interrupted=*/false);
  }
  print_diagnostics();
  if (!write_report()) {
    return fail("writing the report failed: " + vm["report"].as<std::string>());
  }
  if (!pf.silent()) {
    std::string names;
    for (const auto &target : result.targets) {
      if (!names.empty()) {
        names += ",";
      }
      names += target.name;
    }
    std::cerr << "wrote " << out_path << " (" << result.bytes.size() << " bytes)"
              << " model=" << result.model_name << " targets=" << result.targets.size() << " (" << names << ")\n"
              << "  source: " << result.files.size() << " file(s) sha256=" << result.source_sha256.substr(0, 16)
              << " c" << result.level << "\n"
              << "  lowered: " << result.lowering.source_gates << " gates, " << result.lowering.source_basic_events
              << " basic events, " << result.lowering.source_house_events << " house events, "
              << result.lowering.overridden_elements << " overrides\n"
              << "  circuit: nodes " << result.nodes_before << " -> " << result.nodes_after << ", edges "
              << result.edges_before << " -> " << result.edges_after << "\n"
              << "  diagnostics: " << warnings << " warning(s), " << notes << " note(s)\n";
  }
  if (progress_log) {
    fuzzybits::runtime::progress::finish_log(*progress_log);
    if (!progress_log->ok()) {
      return fail("writing the progress log failed: " + progress_log->path().string());
    }
  }
  return 0;
}

int cmd_compile(int argc, char **argv) {
  po::options_description desc("fzb compile options");
  desc.add_options()
      ("help,h", "display this help message")
      ("output,o", po::value<std::string>(), "output .pcfb path")
      ("compilation-passes,c", po::value<int>()->default_value(2),
       "MEF: 0=off 1=null-only 2=optimize 4-8=multipass. FTP: 0=none 1=ir::pc passes (default 1)")
      ("no-kn", "expand k/n to and/or")
      ("no-xor", "expand xor to and/or")
      ("recognize-atleast", "recognize expanded threshold groups (default on)")
      ("no-recognize-atleast", "disable k/n group recognition")
      ("recognize-xor", "recognize expanded xor (default on)")
      ("no-recognize-xor", "disable xor group recognition")
      ("keep-null", "keep NULL/pass-through gates")
      ("ccf", "expand common-cause failure groups")
      ("allow-extern", "**UNSAFE** allow external libraries")
      ("target", po::value<std::vector<std::string>>(),
       "MEF: top-event / gate / nombrilo name. FTP: PROCESS target, gate or basic event "
       "(repeatable; default: every PROCESS target)")
      ("watch", po::value<std::vector<std::string>>(),
       "extra gate / basic event written as an additional query (repeatable)")
      ("override", po::value<std::vector<std::string>>(), "FTP: force an element, NAME=0 or NAME=1 (repeatable)")
      ("duplicate-import", po::value<std::string>(),
       "FTP: event imported more than once: error|first|last (default last)")
      ("complement-events", po::value<std::string>(),
       "FTP: -EVENT operands: atom|negation (default atom: marked literals)")
      ("negation", po::value<std::string>(),
       "FTP: boolean|hull (default hull when the model records ftp.negation = delete_term)")
      ("max-initiators-per-cutset", po::value<std::string>(),
       "FTP: at most N initiating events per cut set, or none (default: the model's "
       "ftp.max_initiators_per_cutset)")
      ("literals", po::value<std::string>(), "FTP hull targets: atom_split|omega_l|negation (default atom_split)")
      ("config", po::value<std::string>(), "FTP: quantification config FILE (JSON); flags override it")
      ("report", po::value<std::string>(), "FTP: write the lowering and pass reports and diagnostics to FILE (JSON)")
      ("weights", po::value<std::string>(), "nombrilo variant weights file")
      ("progress-log", po::value<std::string>(), kProgressLogHelp)
      ("input-file", po::value<std::vector<std::string>>(),
       "MEF XML, nombrilo.circuit, or CAFTA FTP files / a directory of *.ftp files");
  add_progress_options(desc);

  po::positional_options_description pos;
  pos.add("input-file", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb compile model.xml|model.safetensors [-c N] [--no-kn] "
                 "[--target NAME] [--weights FILE] [-o model.pcfb]\n"
                 "       fzb compile a.ftp [b.ftp ...]|DIR [--target NAME]... [--negation boolean|hull] "
                 "[--override NAME=0|1]... [--config FILE] [--report FILE] [-o model.pcfb]\n"
                 "                   [--no-progress] [--quiet]\n\n"
              << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb compile: missing input file\n\n" << desc << std::endl;
    return 1;
  }

  const auto files = vm["input-file"].as<std::vector<std::string>>();
  // Opt-in and independent of the display; opened before lowering starts.
  std::unique_ptr<fuzzybits::runtime::progress::log_sink> progress_log;
  try {
    progress_log = open_progress_log(vm);
  } catch (const std::exception &ex) {
    std::cerr << "fzb compile: " << ex.what() << std::endl;
    return 1;
  }
  const progress_flags pf = read_progress_flags(vm);

  const std::vector<std::filesystem::path> paths(files.begin(), files.end());
  switch (fuzzybits::compile::ftp::classify_inputs(paths)) {
    case fuzzybits::compile::ftp::input_kind::ftp:
      return compile_ftp_inputs(vm, files, pf, progress_log);
    case fuzzybits::compile::ftp::input_kind::mixed:
      std::cerr << "fzb compile: FTP and non-FTP inputs cannot be combined\n";
      return 1;
    case fuzzybits::compile::ftp::input_kind::other:
      break;
  }
  for (const char *flag : {"override", "duplicate-import", "complement-events", "negation", "max-initiators-per-cutset",
                           "literals", "config", "report"}) {
    if (vm.count(flag)) {
      std::cerr << "fzb compile: --" << flag << " applies to CAFTA FTP input only\n";
      return 1;
    }
  }

  fuzzybits::compile::Options opt;
  opt.compilation_level = vm["compilation-passes"].as<int>();
  opt.expand_atleast_gates = vm.count("no-kn") > 0;
  opt.expand_xor_gates = vm.count("no-xor") > 0;
  if (vm.count("no-recognize-atleast")) {
    opt.recognize_atleast_groups = false;
  } else if (vm.count("recognize-atleast")) {
    opt.recognize_atleast_groups = true;
  }
  if (vm.count("no-recognize-xor")) {
    opt.recognize_xor_groups = false;
  } else if (vm.count("recognize-xor")) {
    opt.recognize_xor_groups = true;
  }
  opt.keep_null_gates = vm.count("keep-null") > 0;
  opt.ccf = vm.count("ccf") > 0;
  opt.allow_extern = vm.count("allow-extern") > 0;
  if (vm.count("target")) {
    const auto targets = vm["target"].as<std::vector<std::string>>();
    if (targets.size() > 1) {
      std::cerr << "fzb compile: OpenPSA/MEF input takes one --target\n";
      return 1;
    }
    opt.target = targets.front();
  }
  if (vm.count("watch")) {
    opt.watch = vm["watch"].as<std::vector<std::string>>();
  }
  if (vm.count("weights")) {
    opt.weights_file = vm["weights"].as<std::string>();
  }

  auto progress_sink = pf.attach(opt.progress_sinks);
  if (progress_log) {
    opt.progress_sinks.push_back(progress_log.get());
  }
  auto compiled = fuzzybits::compile::compile_files(files, opt);
  if (!compiled) {
    if (progress_sink) {
      progress_sink->stop();
    }
    std::cerr << "fzb compile: " << compiled.error().message << std::endl;
    return 1;
  }

  std::filesystem::path out_path;
  if (vm.count("output")) {
    out_path = vm["output"].as<std::string>();
  } else {
    out_path = std::filesystem::path(files.front()).stem();
    out_path += ".pcfb";
  }
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    if (progress_sink) {
      progress_sink->stop();
    }
    std::cerr << "fzb compile: cannot write " << out_path << std::endl;
    return 1;
  }
  out.write(reinterpret_cast<const char *>(compiled->bytes.data()),
            static_cast<std::streamsize>(compiled->bytes.size()));
  if (!out) {
    if (progress_sink) {
      progress_sink->stop();
    }
    std::cerr << "fzb compile: write failed: " << out_path << std::endl;
    return 1;
  }
  if (progress_sink) {
    // Summarise before the `wrote …` line so stderr reads in completion order.
    progress_sink->finish_now(/*interrupted=*/false);
  }
  if (!pf.silent()) {
    std::cerr << "wrote " << out_path << " (" << compiled->bytes.size() << " bytes)"
              << " model=" << compiled->model_name << " top=" << compiled->target_id << std::endl;
  }
  if (progress_log) {
    // A compile has no query results; the finishing snapshot carries the stage.
    fuzzybits::runtime::progress::finish_log(*progress_log);
    if (!progress_log->ok()) {
      std::cerr << "fzb compile: writing the progress log failed: " << progress_log->path().string()
                << std::endl;
      return 1;
    }
  }
  return 0;
}

int cmd_inspect(int argc, char **argv) {
  po::options_description desc("fzb inspect options");
  desc.add_options()
      ("help,h", "display this help message")
      ("validate", "run pcfb::validate; exit non-zero on diagnostics")
      ("json", "print summary as JSON")
      ("exact", "exhaustive exact P(query) for small models (≤ 30 events)")
      ("input-file", po::value<std::string>(), ".pcfb file");

  po::positional_options_description pos;
  pos.add("input-file", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb inspect model.pcfb [--validate] [--json] [--exact]\n\n" << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb inspect: missing .pcfb file\n\n" << desc << std::endl;
    return 1;
  }

  const auto path = vm["input-file"].as<std::string>();
  auto circuit = pcfb::io::read(path);
  if (!circuit) {
    std::cerr << "fzb inspect: " << circuit.error().message << std::endl;
    return 1;
  }

  int rc = 0;
  if (vm.count("validate")) {
    const auto diags = pcfb::validate(*circuit);
    if (!diags.empty()) {
      rc = 1;
      for (const auto &d : diags) {
        std::cerr << "validate: " << d.location << ": " << d.message << std::endl;
      }
    } else if (!vm.count("json")) {
      std::cerr << "validate: ok\n";
    }
  }

  const auto summary = summarise(*circuit);
  if (vm.count("json")) {
    print_json(summary, std::cout);
  } else {
    print_text(summary, std::cout);
  }

  if (vm.count("exact")) {
    try {
      const double p = fuzzybits::compile::exact_probability(*circuit);
      std::cout << std::setprecision(16) << std::scientific << "exact_probability  " << p << std::endl;
    } catch (const fuzzybits::compile::ExactEvalError &ex) {
      std::cerr << "fzb inspect --exact: " << ex.what() << std::endl;
      return 1;
    }
  }
  return rc;
}

int cmd_export(int argc, char **argv) {
  po::options_description desc("fzb export options");
  desc.add_options()
      ("help,h", "display this help message")
      ("to", po::value<std::string>(), "target format (nombrilo)")
      ("output,o", po::value<std::string>(), "output path")
      ("strict", "omit fuzzybits.pcfb.* extras (plain nombrilo)")
      ("input-file", po::value<std::string>(), ".pcfb file");

  po::positional_options_description pos;
  pos.add("input-file", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb export --to nombrilo model.pcfb -o out.safetensors [--strict]\n\n"
              << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb export: missing .pcfb file\n\n" << desc << std::endl;
    return 1;
  }
  const std::string to = vm.count("to") ? vm["to"].as<std::string>() : std::string();
  if (to != "nombrilo" && to != "nombrilo.circuit") {
    std::cerr << "fzb export: --to nombrilo is required\n\n" << desc << std::endl;
    return 1;
  }

  auto circuit = pcfb::io::read(vm["input-file"].as<std::string>());
  if (!circuit) {
    std::cerr << "fzb export: " << circuit.error().message << std::endl;
    return 1;
  }
  fuzzybits::compile::NombriloExportOptions opt;
  opt.extras = vm.count("strict") == 0;
  auto nombrilo = fuzzybits::compile::pcfb_to_nombrilo(*circuit, opt);
  if (!nombrilo) {
    std::cerr << "fzb export: " << nombrilo.error().message << std::endl;
    return 1;
  }
  auto bytes = fuzzybits::compile::write_nombrilo(*nombrilo);
  if (!bytes) {
    std::cerr << "fzb export: " << bytes.error().message << std::endl;
    return 1;
  }

  std::filesystem::path out_path;
  if (vm.count("output")) {
    out_path = vm["output"].as<std::string>();
  } else {
    out_path = std::filesystem::path(vm["input-file"].as<std::string>()).stem();
    out_path += ".safetensors";
  }
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "fzb export: cannot write " << out_path << std::endl;
    return 1;
  }
  out.write(reinterpret_cast<const char *>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
  if (!out) {
    std::cerr << "fzb export: write failed: " << out_path << std::endl;
    return 1;
  }
  std::cerr << "wrote " << out_path << " (" << bytes->size() << " bytes)" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    print_root_usage(std::cerr);
    return 1;
  }
  const std::string cmd = argv[1];
  if (cmd == "-h" || cmd == "--help" || cmd == "help") {
    print_root_usage(std::cout);
    return 0;
  }
  if (cmd == "-v" || cmd == "--version" || cmd == "version") {
    std::cout << "fzb " << SCRAM_VERSION << " (" << SCRAM_GIT_REVISION << ")\n";
    return 0;
  }
  if (cmd == "compile") {
    return cmd_compile(argc - 1, argv + 1);
  }
  if (cmd == "export") {
    return cmd_export(argc - 1, argv + 1);
  }
  if (cmd == "inspect") {
    return cmd_inspect(argc - 1, argv + 1);
  }
  if (cmd == "run") {
    return cmd_run(argc - 1, argv + 1);
  }
  std::cerr << "fzb: unknown command '" << cmd << "'\n";
  print_root_usage(std::cerr);
  return 1;
}

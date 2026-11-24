#include "compile/ftp/compile.h"

#include "compile/ftp/json.h"
#include "compile/pra_to_pc/compose.h"
#include "compile/pra_to_pc/descriptor.h"
#include "compile/sha256.h"
#include "io/pcfb/descriptor.h"
#include "io/pcfb/frequency.h"
#include "ir/pc/circuit.h"
#include "ir/pc/passes/passes.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fuzzybits::compile::ftp {

namespace {

namespace pra = ir::pra;
namespace pc = ir::pc;
namespace reader = io::backend::fuzzybits::ftp;
namespace lowering_ns = backend::fuzzybits::pra_to_pc;
namespace prog = runtime::progress;
namespace pcfb_frequency = io::pcfb_frequency;
using clock = std::chrono::steady_clock;

constexpr std::string_view source_format = "cafta-ftp";
constexpr std::string_view max_initiators_attribute = "ftp.max_initiators_per_cutset";
constexpr std::string_view limit_attribute = "ftp.limit";

[[nodiscard]] auto seconds_since(const clock::time_point start) -> double {
    return std::chrono::duration<double>(clock::now() - start).count();
}

[[nodiscard]] auto lower_case(std::string text) -> std::string {
    std::ranges::transform(text, text.begin(),
                           [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

[[nodiscard]] auto in_quotes(std::string_view text) -> std::string { return "'" + std::string(text) + "'"; }

[[nodiscard]] auto option_error(std::string message) -> api::diagnostic {
    return api::make_error("compile.ftp.invalid-option", std::move(message));
}

// Compile-stage snapshots, as the MEF path emits them.
class progress_reporter final {
  public:
    explicit progress_reporter(options const &settings)
        : emitter_(settings.progress_sinks, settings.progress_max_rate_hz), start_(clock::now()) {}

    void stage(std::string step) {
        if (!emitter_.active()) {
            return;
        }
        step_ = step;
        prog::snapshot snapshot;
        snapshot.stage = prog::phase::compile;
        snapshot.step = std::move(step);
        snapshot.elapsed_seconds = seconds_since(start_);
        emitter_.emit_now(std::move(snapshot));
    }

    void finish(const bool interrupted) {
        if (!emitter_.active() || finished_) {
            return;
        }
        finished_ = true;
        prog::snapshot last;
        last.stage = interrupted ? prog::phase::compile : prog::phase::done;
        last.step = interrupted ? step_ : std::string{};
        last.elapsed_seconds = seconds_since(start_);
        if (!interrupted) {
            emitter_.finish(std::move(last), false);
            return;
        }
        try {
            emitter_.finish(std::move(last), true);
        } catch (...) {
        }
    }

  private:
    prog::emitter emitter_;
    clock::time_point start_;
    std::string step_;
    bool finished_ = false;
};

// ---- Settings -------------------------------------------------------

[[nodiscard]] auto parse_negation(std::string_view text) -> std::optional<pra::negation_set> {
    if (text == "boolean") {
        return pra::negation_set::of(pra::negation_semantics::boolean);
    }
    if (text == "hull" || text == "monotone_hull") {
        return pra::negation_set::of(pra::negation_semantics::monotone_hull);
    }
    if (text == "both") {
        return pra::negation_set::both();
    }
    return std::nullopt;
}

[[nodiscard]] auto parse_literals(std::string_view text) -> std::optional<pra::literal_mode> {
    if (text == "atom_split") {
        return pra::literal_mode::atom_split;
    }
    if (text == "omega_l") {
        return pra::literal_mode::omega_l;
    }
    if (text == "negation") {
        return pra::literal_mode::negation;
    }
    return std::nullopt;
}

[[nodiscard]] auto literals_name(const pra::literal_mode mode) -> std::string_view {
    switch (mode) {
    case pra::literal_mode::atom_split:
        return "atom_split";
    case pra::literal_mode::omega_l:
        return "omega_l";
    case pra::literal_mode::negation:
        return "negation";
    }
    return "atom_split";
}

[[nodiscard]] auto negation_json(const pra::negation_set set) -> json::value {
    if (set.boolean && set.monotone_hull) {
        return json::value::make_array()
            .push(json::value::make_string("boolean"))
            .push(json::value::make_string("hull"));
    }
    return json::value::make_string(set.monotone_hull ? "hull" : "boolean");
}

// One target of the `--config` document or of the command line.
struct target_spec final {
    std::string name;
    // domain_target | gate | basic_event; unset: look the name up.
    std::optional<std::string> subject;
    std::optional<pra::negation_set> negation;
    std::optional<pra::literal_mode> literals;
};

struct override_item final {
    std::string name;
    bool value = false;
    std::optional<std::string> kind;
};

struct settings_document final {
    std::optional<std::vector<target_spec>> targets;
    std::optional<std::vector<target_spec>> watch;
    std::vector<override_item> overrides;
    std::optional<pra::negation_set> negation;
    std::optional<pra::literal_mode> literals;
    std::optional<std::string> duplicate_import;
    std::optional<std::string> complement_events;
    std::optional<std::optional<std::uint32_t>> max_initiators;
    std::optional<std::string> initiator_values;
    std::optional<std::string> initiator_unit;
    std::optional<int> level;
};

class document_reader final {
  public:
    explicit document_reader(api::diagnostic_list &diagnostics) : diagnostics_(diagnostics) {}

    [[nodiscard]] auto read(json::value const &root) -> std::optional<settings_document> {
        if (!root.is_object()) {
            return fail("the document must be a JSON object");
        }
        static constexpr std::array<std::string_view, 11> known{"targets",           "watch",
                                                                "overrides",         "negation",
                                                                "literals",          "duplicate_import",
                                                                "complement_events", "max_initiators_per_cutset",
                                                                "initiator_values",  "initiator_unit",
                                                                "compilation_level"};
        for (auto const &key : root.keys) {
            if (std::ranges::find(known, std::string_view(key)) == known.end()) {
                return fail("unknown key " + in_quotes(key));
            }
        }
        settings_document out;
        if (auto const *value = root.find("targets")) {
            out.targets = targets(*value, "targets");
        }
        if (auto const *value = root.find("watch")) {
            out.watch = targets(*value, "watch");
        }
        if (auto const *value = root.find("overrides")) {
            overrides(*value, out.overrides);
        }
        if (auto const *value = root.find("negation")) {
            out.negation = negation(*value, "negation");
        }
        if (auto const *value = root.find("literals")) {
            out.literals = literals(*value, "literals");
        }
        if (auto const *value = root.find("duplicate_import")) {
            out.duplicate_import = choice(*value, "duplicate_import", {"error", "first", "last"});
        }
        if (auto const *value = root.find("complement_events")) {
            out.complement_events = choice(*value, "complement_events", {"atom", "negation"});
        }
        if (auto const *value = root.find("max_initiators_per_cutset")) {
            if (value->is(json::value::kind_type::null) || (value->is_string() && value->string == "none")) {
                out.max_initiators = std::optional<std::uint32_t>{};
            } else if (value->is_integer() && value->integer >= 0 && value->integer <= 0xFFFFFFFFLL) {
                out.max_initiators = std::optional<std::uint32_t>{static_cast<std::uint32_t>(value->integer)};
            } else {
                error("max_initiators_per_cutset must be a non-negative integer, null or \"none\"");
            }
        }
        if (auto const *value = root.find("initiator_values")) {
            out.initiator_values = choice(*value, "initiator_values", {"frequency", "probability"});
        }
        if (auto const *value = root.find("initiator_unit")) {
            if (value->is_string() && pcfb_frequency::find_unit(value->string)) {
                out.initiator_unit = value->string;
            } else {
                error("initiator_unit must be one of " + pcfb_frequency::unit_names());
            }
        }
        if (auto const *value = root.find("compilation_level")) {
            if (value->is_integer() && (value->integer == 0 || value->integer == 1)) {
                out.level = static_cast<int>(value->integer);
            } else {
                error("compilation_level must be 0 or 1");
            }
        }
        if (failed_) {
            return std::nullopt;
        }
        return out;
    }

  private:
    auto fail(std::string message) -> std::optional<settings_document> {
        error(std::move(message));
        return std::nullopt;
    }

    void error(std::string message) {
        failed_ = true;
        diagnostics_.push(api::make_error("compile.ftp.invalid-config", std::move(message)));
    }

    [[nodiscard]] auto choice(json::value const &value, std::string_view key,
                              std::initializer_list<std::string_view> allowed) -> std::optional<std::string> {
        if (value.is_string() && std::ranges::find(allowed, std::string_view(value.string)) != allowed.end()) {
            return value.string;
        }
        std::string list;
        for (const auto item : allowed) {
            list += (list.empty() ? "" : ", ") + std::string(item);
        }
        error(std::string(key) + " must be one of " + list);
        return std::nullopt;
    }

    [[nodiscard]] auto negation(json::value const &value, std::string_view key) -> std::optional<pra::negation_set> {
        std::optional<pra::negation_set> out;
        if (value.is_string()) {
            out = parse_negation(value.string);
        } else if (value.is_array() && !value.items.empty()) {
            pra::negation_set set{.boolean = false, .monotone_hull = false};
            bool valid = true;
            for (auto const &item : value.items) {
                const auto one = item.is_string() ? parse_negation(item.string) : std::nullopt;
                if (!one || (one->boolean && one->monotone_hull)) {
                    valid = false;
                    break;
                }
                set.boolean = set.boolean || one->boolean;
                set.monotone_hull = set.monotone_hull || one->monotone_hull;
            }
            if (valid) {
                out = set;
            }
        }
        if (!out) {
            error(std::string(key) +
                  " must be \"boolean\", \"hull\", \"both\" or an array of \"boolean\" and \"hull\"");
        }
        return out;
    }

    [[nodiscard]] auto literals(json::value const &value, std::string_view key) -> std::optional<pra::literal_mode> {
        const auto out = value.is_string() ? parse_literals(value.string) : std::nullopt;
        if (!out) {
            error(std::string(key) + " must be \"atom_split\", \"omega_l\" or \"negation\"");
        }
        return out;
    }

    [[nodiscard]] auto targets(json::value const &list, std::string_view key) -> std::vector<target_spec> {
        std::vector<target_spec> out;
        if (!list.is_array()) {
            error(std::string(key) + " must be an array");
            return out;
        }
        for (auto const &item : list.items) {
            target_spec spec;
            if (item.is_string()) {
                spec.name = item.string;
            } else if (item.is_object()) {
                for (auto const &member : item.keys) {
                    if (member != "name" && member != "subject" && member != "negation" && member != "literals") {
                        error(std::string(key) + " entry has unknown key " + in_quotes(member));
                    }
                }
                if (auto const *name = item.find("name"); name != nullptr && name->is_string()) {
                    spec.name = name->string;
                } else {
                    error(std::string(key) + " entry needs a string \"name\"");
                    continue;
                }
                if (auto const *subject = item.find("subject")) {
                    if (subject->is_string() && (subject->string == "domain_target" || subject->string == "gate" ||
                                                 subject->string == "basic_event")) {
                        spec.subject = subject->string;
                    } else {
                        error("subject must be \"domain_target\", \"gate\" or \"basic_event\"");
                    }
                }
                if (auto const *negation_value = item.find("negation")) {
                    spec.negation = negation(*negation_value, "negation");
                }
                if (auto const *literals_value = item.find("literals")) {
                    spec.literals = literals(*literals_value, "literals");
                }
            } else {
                error(std::string(key) + " entries must be names or objects");
                continue;
            }
            out.push_back(std::move(spec));
        }
        return out;
    }

    static auto flag_value(json::value const &value) -> std::optional<bool> {
        if (value.is(json::value::kind_type::boolean)) {
            return value.boolean;
        }
        if (value.is_integer() && (value.integer == 0 || value.integer == 1)) {
            return value.integer == 1;
        }
        return std::nullopt;
    }

    void overrides(json::value const &list, std::vector<override_item> &out) {
        if (list.is_object()) {
            for (std::size_t index = 0; index < list.keys.size(); ++index) {
                if (const auto value = flag_value(list.items[index])) {
                    out.push_back(override_item{.name = list.keys[index], .value = *value});
                } else {
                    error("override " + in_quotes(list.keys[index]) + " must be true, false, 1 or 0");
                }
            }
            return;
        }
        if (!list.is_array()) {
            error("overrides must be an object of name: value or an array of {name, value}");
            return;
        }
        for (auto const &item : list.items) {
            auto const *name = item.find("name");
            auto const *value = item.find("value");
            const auto flag = value != nullptr ? flag_value(*value) : std::nullopt;
            if (name == nullptr || !name->is_string() || !flag) {
                error("override entries need a string \"name\" and a boolean \"value\"");
                continue;
            }
            override_item entry{.name = name->string, .value = *flag};
            if (auto const *kind = item.find("kind")) {
                const bool known = kind->is_string() && (kind->string == "gate" || kind->string == "basic_event" ||
                                                         kind->string == "house_event");
                if (!known) {
                    error("override kind must be \"gate\", \"basic_event\" or \"house_event\"");
                    continue;
                }
                entry.kind = kind->string;
            }
            out.push_back(std::move(entry));
        }
    }

    api::diagnostic_list &diagnostics_;
    bool failed_ = false;
};

// The settings that survive merging the flags, the document and the model.
struct effective final {
    std::string duplicate_import = "last";
    std::string complement_events = "atom";
    pra::negation_set negation;
    pra::literal_mode literals = pra::literal_mode::atom_split;
    std::optional<std::uint32_t> max_initiators;
    std::string initiator_values = "frequency";
    std::optional<std::string> initiator_unit;
    int level = 1;
    std::vector<override_item> overrides;
};

struct source_set final {
    std::vector<reader::source_text> sources;
    std::string sha256_hex;
    std::array<std::byte, 32> digest{};
};

// The files, sorted by path and without repeats, and their digest: the
// SHA-256 of the contents concatenated in that order.
[[nodiscard]] auto sorted_sources(std::span<reader::source_text const> given) -> source_set {
    source_set out;
    out.sources.assign(given.begin(), given.end());
    std::ranges::stable_sort(out.sources, {}, &reader::source_text::path);
    out.sources.erase(std::ranges::unique(out.sources, {}, &reader::source_text::path).begin(), out.sources.end());
    std::vector<std::byte> bytes;
    for (auto const &source : out.sources) {
        const auto *begin = reinterpret_cast<std::byte const *>(source.text.data());
        bytes.insert(bytes.end(), begin, begin + source.text.size());
    }
    const auto digest = compile::sha256(bytes);
    for (std::size_t index = 0; index < digest.size(); ++index) {
        out.digest[index] = static_cast<std::byte>(digest[index]);
    }
    out.sha256_hex = compile::sha256_hex(digest);
    return out;
}

[[nodiscard]] auto attribute_value(pra::model const &model, std::string_view name) -> std::optional<std::string> {
    for (auto const &attribute : model.project_attributes()) {
        if (model.text(attribute.name) == name) {
            return std::string(model.text(attribute.value));
        }
    }
    return std::nullopt;
}

// ---- Lowering stage -------------------------------------------------

struct lowered_stage final {
    lowering_ns::pra_pc_lowering lowering;
    effective settings;
    std::string limit;
};

[[nodiscard]] auto resolve_request(pra::model const &model, target_spec const &spec,
                                   std::optional<pra::negation_set> negation_flag,
                                   std::optional<pra::literal_mode> literals_flag, pra::negation_set default_negation,
                                   api::diagnostic_list &diagnostics) -> std::optional<lowering_ns::target_request> {
    lowering_ns::target_request request{.name = spec.name,
                                        .negation = negation_flag.value_or(spec.negation.value_or(default_negation))};
    request.literals = literals_flag ? literals_flag : spec.literals;
    std::string subject;
    if (spec.subject) {
        subject = *spec.subject;
    } else if (model.find_target(spec.name).valid()) {
        subject = "domain_target";
    } else if (model.find_gate(spec.name).valid()) {
        subject = "gate";
    } else if (model.find_basic_event(spec.name).valid()) {
        subject = "basic_event";
    } else {
        diagnostics.push(api::make_error("compile.ftp.unknown-target",
                                         "no PROCESS target, gate or basic event is named " + in_quotes(spec.name)));
        return std::nullopt;
    }
    if (subject == "gate") {
        request.element = pra::element_kind::gate;
    } else if (subject == "basic_event") {
        request.element = pra::element_kind::basic_event;
    }
    return request;
}

[[nodiscard]] auto run_lower(std::span<reader::source_text const> given, options const &settings,
                             api::diagnostic_list &diagnostics, timings &seconds, progress_reporter *progress,
                             source_set &set) -> std::optional<lowered_stage> {
    if (settings.level && *settings.level != 0 && *settings.level != 1) {
        diagnostics.push(option_error("-c must be 0 or 1 for FTP input, not " + std::to_string(*settings.level)));
        return std::nullopt;
    }
    for (auto const &entry : settings.overrides) {
        if (entry.name.empty()) {
            diagnostics.push(option_error("an override has an empty name"));
            return std::nullopt;
        }
    }
    if (settings.negation && !parse_negation(*settings.negation)) {
        diagnostics.push(option_error("negation must be boolean, hull or both, not " + in_quotes(*settings.negation)));
    }
    if (settings.literals && !parse_literals(*settings.literals)) {
        diagnostics.push(
            option_error("literals must be atom_split, omega_l or negation, not " + in_quotes(*settings.literals)));
    }
    if (settings.duplicate_import && *settings.duplicate_import != "error" && *settings.duplicate_import != "first" &&
        *settings.duplicate_import != "last") {
        diagnostics.push(option_error("duplicate-import must be error, first or last, not " +
                                      in_quotes(*settings.duplicate_import)));
    }
    if (settings.complement_events && *settings.complement_events != "atom" &&
        *settings.complement_events != "negation") {
        diagnostics.push(
            option_error("complement-events must be atom or negation, not " + in_quotes(*settings.complement_events)));
    }
    if (settings.initiator_values && *settings.initiator_values != "frequency" &&
        *settings.initiator_values != "probability") {
        diagnostics.push(option_error("initiator-values must be frequency or probability, not " +
                                      in_quotes(*settings.initiator_values)));
    }
    if (settings.initiator_unit && !pcfb_frequency::find_unit(*settings.initiator_unit)) {
        diagnostics.push(option_error("initiator-unit must be one of " + pcfb_frequency::unit_names() + ", not " +
                                      in_quotes(*settings.initiator_unit)));
    }
    settings_document document;
    if (settings.config) {
        auto parsed = json::parse(*settings.config);
        if (!parsed) {
            diagnostics.append(parsed.diagnostics());
            return std::nullopt;
        }
        auto read = document_reader(diagnostics).read(parsed.value());
        if (!read) {
            return std::nullopt;
        }
        document = std::move(*read);
    }
    if (diagnostics.has_errors()) {
        return std::nullopt;
    }

    effective merged;
    merged.duplicate_import = settings.duplicate_import.value_or(document.duplicate_import.value_or("last"));
    merged.complement_events = settings.complement_events.value_or(document.complement_events.value_or("atom"));
    merged.level = settings.level.value_or(document.level.value_or(1));
    if (merged.level != 0 && merged.level != 1) {
        diagnostics.push(option_error("-c must be 0 or 1 for FTP input, not " + std::to_string(merged.level)));
        return std::nullopt;
    }

    if (progress != nullptr) {
        progress->stage("parse");
    }
    const auto read_start = clock::now();
    set = sorted_sources(given);
    api::option_set reader_options;
    reader_options.set(reader::duplicate_import_option(), merged.duplicate_import);
    reader_options.set(reader::complement_events_option(), merged.complement_events);
    auto project = reader::read_project(std::span<reader::source_text const>(set.sources), reader_options);
    diagnostics.append(project.diagnostics());
    seconds.read = seconds_since(read_start);
    if (!project) {
        return std::nullopt;
    }
    auto const *model = pra::model_from(project.value());
    if (model == nullptr) {
        diagnostics.push(
            api::make_error("compile.ftp.invalid-model", "the FTP reader did not produce an ir::pra model"));
        return std::nullopt;
    }

    // The query type is requested, never read from the model: every target is a
    // probability query unless the flags or the document ask for the hull.
    if (settings.negation) {
        merged.negation = *parse_negation(*settings.negation);
    } else {
        merged.negation = document.negation.value_or(pra::negation_set::of(pra::negation_semantics::boolean));
    }
    merged.literals = settings.literals ? *parse_literals(*settings.literals)
                                        : document.literals.value_or(pra::literal_mode::atom_split);

    merged.initiator_values =
        settings.initiator_values.value_or(document.initiator_values.value_or(std::string("frequency")));
    merged.initiator_unit = settings.initiator_unit ? settings.initiator_unit : document.initiator_unit;
    const bool frequencies = merged.initiator_values == "frequency";
    // `none` (no rule) agrees with frequencies; a count is a rule, which they make redundant.
    const auto rule_count = [](auto const &setting) { return setting && setting->has_value(); };
    if (frequencies && (rule_count(settings.max_initiators_per_cutset) || rule_count(document.max_initiators))) {
        diagnostics.push(option_error("max-initiators-per-cutset N applies only with --initiator-values probability: "
                                      "with frequencies each conditioned query has exactly one initiating event"));
        return std::nullopt;
    }
    if (!frequencies && merged.initiator_unit) {
        diagnostics.push(option_error("initiator-unit applies only with --initiator-values frequency"));
        return std::nullopt;
    }
    if (frequencies && !merged.initiator_unit) {
        merged.initiator_unit = std::string(pcfb_frequency::default_unit);
    }

    bool initiators_from_attribute = false;
    if (frequencies) {
        // The attribute records the cut-set convention of at most one initiator, which the
        // conditioning already implies.
    } else if (settings.max_initiators_per_cutset) {
        merged.max_initiators = *settings.max_initiators_per_cutset;
    } else if (document.max_initiators) {
        merged.max_initiators = *document.max_initiators;
    } else if (const auto attribute = attribute_value(*model, max_initiators_attribute)) {
        std::uint32_t count = 0;
        const auto *end = attribute->data() + attribute->size();
        if (const auto [pointer, status] = std::from_chars(attribute->data(), end, count);
            status == std::errc{} && pointer == end) {
            merged.max_initiators = count;
            initiators_from_attribute = true;
        } else if (*attribute != "none") {
            diagnostics.push(api::make_warning("compile.ftp.unknown-attribute",
                                               "model attribute ftp.max_initiators_per_cutset has the unknown value " +
                                                   in_quotes(*attribute) + "; applying no initiator rule"));
        }
    }
    if (initiators_from_attribute) {
        diagnostics.push(
            api::make_note("compile.ftp.default-initiator-rule",
                           "at most " + std::to_string(*merged.max_initiators) +
                               " initiating event(s) per cut set: the model records "
                               "ftp.max_initiators_per_cutset (--max-initiators-per-cutset none disables the rule)"));
    }

    lowering_ns::quantification_config config;
    config.default_negation = merged.negation;
    config.default_literals = merged.literals;
    config.max_initiators_per_cutset = merged.max_initiators;
    config.initiators =
        frequencies ? lowering_ns::initiator_values::frequency : lowering_ns::initiator_values::probability;
    config.initiator_unit = merged.initiator_unit;

    const std::optional<pra::negation_set> negation_flag =
        settings.negation ? std::optional<pra::negation_set>(merged.negation) : std::nullopt;
    const std::optional<pra::literal_mode> literals_flag =
        settings.literals ? std::optional<pra::literal_mode>(merged.literals) : std::nullopt;
    const auto add = [&](std::vector<lowering_ns::target_request> &into, target_spec const &spec) {
        if (auto request = resolve_request(*model, spec, negation_flag, literals_flag, merged.negation, diagnostics)) {
            into.push_back(std::move(*request));
        }
    };
    std::vector<target_spec> targets;
    if (!settings.targets.empty()) {
        for (auto const &name : settings.targets) {
            target_spec spec{.name = name};
            if (document.targets) {
                if (const auto found = std::ranges::find(*document.targets, name, &target_spec::name);
                    found != document.targets->end()) {
                    spec = *found;
                }
            }
            targets.push_back(std::move(spec));
        }
    } else if (document.targets) {
        targets = *document.targets;
    }
    for (auto const &spec : targets) {
        add(config.targets, spec);
    }
    std::vector<target_spec> watch;
    if (!settings.watch.empty()) {
        for (auto const &name : settings.watch) {
            target_spec spec{.name = name};
            if (document.watch) {
                if (const auto found = std::ranges::find(*document.watch, name, &target_spec::name);
                    found != document.watch->end()) {
                    spec = *found;
                }
            }
            watch.push_back(std::move(spec));
        }
    } else if (document.watch) {
        watch = *document.watch;
    }
    for (auto const &spec : watch) {
        add(config.watch_targets, spec);
    }

    merged.overrides = document.overrides;
    for (auto const &entry : settings.overrides) {
        if (const auto found = std::ranges::find(merged.overrides, entry.name, &override_item::name);
            found != merged.overrides.end()) {
            found->value = entry.value;
        } else {
            merged.overrides.push_back(override_item{.name = entry.name, .value = entry.value});
        }
    }
    for (auto const &entry : merged.overrides) {
        lowering_ns::override_request request{.name = entry.name, .value = entry.value};
        if (entry.kind == "gate") {
            request.kind = pra::element_kind::gate;
        } else if (entry.kind == "basic_event") {
            request.kind = pra::element_kind::basic_event;
        } else if (entry.kind == "house_event") {
            request.kind = pra::element_kind::house_event;
        }
        config.overrides.push_back(std::move(request));
    }
    if (diagnostics.has_errors()) {
        return std::nullopt;
    }

    if (progress != nullptr) {
        progress->stage("lower");
    }
    const auto lower_start = clock::now();
    auto lowered = lowering_ns::lower(project.value(), config);
    diagnostics.append(lowered.diagnostics());
    seconds.lower = seconds_since(lower_start);
    if (!lowered) {
        return std::nullopt;
    }
    return lowered_stage{
        .lowering = std::move(lowered).value(),
        .settings = std::move(merged),
        .limit = attribute_value(*model, limit_attribute).value_or(std::string{}),
    };
}

// ---- Summaries ------------------------------------------------------

[[nodiscard]] auto semantics_name(const pc::passes::semantic_class semantics) -> std::string {
    return semantics == pc::passes::semantic_class::function_preserving ? "function_preserving"
                                                                        : "distribution_preserving";
}

[[nodiscard]] auto subject_name(const pra::element_kind kind, const bool domain) -> std::string {
    if (domain) {
        return "domain_target";
    }
    return kind == pra::element_kind::gate ? "gate" : "basic_event";
}

[[nodiscard]] auto summarize_targets(lowered_stage const &stage) -> std::vector<target_summary> {
    auto const *model = pra::model_from(stage.lowering.pra_model);
    std::vector<target_summary> out;
    for (auto const &binding : stage.lowering.projection.target_bindings()) {
        target_summary summary;
        summary.name =
            std::string(binding.domain_target.valid() ? model->stable_name(pra::domain_ref(binding.domain_target))
                                                      : model->stable_name(binding.subject));
        summary.subject = subject_name(binding.subject.kind(), binding.domain_target.valid());
        summary.watch = binding.role == pc::observable_role::watch;
        summary.boolean = binding.negation.boolean;
        summary.monotone_hull = binding.negation.monotone_hull;
        summary.literals = std::string(literals_name(binding.literals));
        summary.max_initiators_per_cutset = stage.settings.max_initiators;
        summary.initiating_events = stage.lowering.projection.initiators(binding).size();
        summary.literal_events = stage.lowering.projection.literal_events(binding).size();
        if (binding.condition.valid()) {
            const pra::domain_ref initiator(binding.condition);
            summary.initiator = std::string(model->stable_name(initiator));
            summary.initiator_frequency =
                model->expressions()[model->basic_events()[binding.condition.raw()].expression.raw()].value;
        }
        out.push_back(std::move(summary));
    }
    return out;
}

[[nodiscard]] auto optional_count(const std::optional<std::uint32_t> value) -> json::value {
    return value ? json::value::make_int(*value) : json::value::make_null();
}

[[nodiscard]] auto options_document(lowered_stage const &stage, std::vector<target_summary> const &targets,
                                    std::string const &limit) -> json::value {
    auto root = json::value::make_object();
    root.set("format", json::value::make_string(std::string(source_format)));
    root.set("compilation_level", json::value::make_int(stage.settings.level));
    root.set("duplicate_import", json::value::make_string(stage.settings.duplicate_import));
    root.set("complement_events", json::value::make_string(stage.settings.complement_events));
    root.set("max_initiators_per_cutset", optional_count(stage.settings.max_initiators));
    root.set("initiator_values", json::value::make_string(stage.settings.initiator_values));
    root.set("initiator_unit", stage.settings.initiator_unit ? json::value::make_string(*stage.settings.initiator_unit)
                                                             : json::value::make_null());
    auto primary = json::value::make_array();
    auto watch = json::value::make_array();
    for (auto const &target : targets) {
        auto entry = json::value::make_object();
        entry.set("name", json::value::make_string(target.name));
        entry.set("subject", json::value::make_string(target.subject));
        entry.set("negation",
                  negation_json(pra::negation_set{.boolean = target.boolean, .monotone_hull = target.monotone_hull}));
        entry.set("literals", json::value::make_string(target.literals));
        entry.set("initiating_events", json::value::make_int(static_cast<std::int64_t>(target.initiating_events)));
        entry.set("literal_events", json::value::make_int(static_cast<std::int64_t>(target.literal_events)));
        if (!target.initiator.empty()) {
            entry.set("initiator", json::value::make_string(target.initiator));
            entry.set("initiator_frequency", json::value::make_number(*target.initiator_frequency));
        }
        (target.watch ? watch : primary).push(std::move(entry));
    }
    root.set("targets", std::move(primary));
    root.set("watch", std::move(watch));
    auto overrides = json::value::make_array();
    for (auto const &entry : stage.settings.overrides) {
        auto item = json::value::make_object();
        item.set("name", json::value::make_string(entry.name));
        item.set("value", json::value::make_bool(entry.value));
        overrides.push(std::move(item));
    }
    root.set("overrides", std::move(overrides));
    if (!limit.empty()) {
        root.set("ftp_limit", json::value::make_string(limit));
    }
    return root;
}

[[nodiscard]] auto model_name_of(std::string_view project_name, std::span<reader::source_text const> sources)
    -> std::string {
    if (!project_name.empty()) {
        return std::string(project_name);
    }
    return sources.empty() ? std::string{} : std::filesystem::path(sources.front().path).stem().string();
}

[[nodiscard]] auto compile_set(std::span<reader::source_text const> given, options const &settings,
                               std::string directory_name) -> compile_result {
    progress_reporter progress(settings);
    api::diagnostic_list diagnostics;
    compiled out;
    source_set set;
    const auto fail = [&](api::diagnostic_list list) {
        progress.finish(true);
        return compile_result::fail(std::move(list));
    };
    try {
        auto stage = run_lower(given, settings, diagnostics, out.seconds, &progress, set);
        if (!stage) {
            return fail(std::move(diagnostics));
        }
        out.level = stage->settings.level;
        out.targets = summarize_targets(*stage);
        out.lowering = stage->lowering.report;
        out.limit = stage->limit;
        out.source_sha256 = set.sha256_hex;
        for (auto const &source : set.sources) {
            out.files.push_back(source.path);
        }

        auto write_options = lowering_ns::pcfb_write_options(stage->lowering);
        if (!write_options) {
            // Raised per target by the lowering; one more message with the ways out.
            bool literals = false;
            for (auto const &item : write_options.diagnostics().items()) {
                literals = literals || item.code == "compile.pra_to_pc.hull-literals-unsupported";
                diagnostics.push(item);
            }
            if (literals) {
                diagnostics.push(api::make_error(
                    "compile.ftp.hull-literals-unsupported",
                    "monotone-hull targets with marked literal operands (-EVENT) cannot be written until pcfb carries "
                    "literal atoms; each alternative is a different query: --negation boolean (the probability, "
                    "the default), or the plain hull with --literals negation or --complement-events negation"));
            }
            return fail(std::move(diagnostics));
        }
        diagnostics.append(write_options.diagnostics());

        progress.stage("passes");
        const auto passes_start = clock::now();
        pc::passes::pipeline_options pipeline;
        auto const bindings = stage->lowering.projection.target_bindings();
        pipeline.semantics.reserve(bindings.size());
        pipeline.atom_events.resize(bindings.size());
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            auto const &binding = bindings[index];
            // A target asking for both kinds takes the stricter pass rules.
            pipeline.semantics.push_back(binding.negation.monotone_hull ? pc::passes::target_semantics::monotone_hull
                                                                        : pc::passes::target_semantics::boolean);
            if (binding.negation.monotone_hull && binding.literals == pra::literal_mode::atom_split) {
                for (const auto event : stage->lowering.projection.literal_events(binding)) {
                    if (const auto node = stage->lowering.projection.node_of(pra::domain_ref(event)); node.valid()) {
                        pipeline.atom_events[index].push_back(node);
                    }
                }
            }
        }
        auto passes = pc::passes::compile(
            stage->lowering.circuit,
            stage->settings.level == 0 ? pc::passes::compile_level::c0 : pc::passes::compile_level::c1, pipeline);
        diagnostics.append(passes.diagnostics);
        out.seconds.passes = seconds_since(passes_start);
        if (!passes.ok()) {
            return fail(std::move(diagnostics));
        }
        out.nodes_before = passes.nodes_before;
        out.edges_before = passes.edges_before;
        out.nodes_after = passes.nodes_after;
        out.edges_after = passes.edges_after;
        for (auto const &report : passes.passes) {
            out.passes.push_back(pass_summary{
                .transform = report.transform_id,
                .semantics = semantics_name(report.semantics),
                .nodes_before = report.nodes_before,
                .edges_before = report.edges_before,
                .nodes_after = report.nodes_after,
                .edges_after = report.edges_after,
                .rewrites = report.rewrites,
                .skipped = report.skipped,
            });
        }

        progress.stage("write");
        const auto write_start = clock::now();
        auto const *model = pra::model_from(stage->lowering.pra_model);
        auto named =
            lowering_ns::compose_write_options(std::move(write_options).value(), stage->lowering.circuit, passes);
        named.model_name = model_name_of(directory_name.empty() ? model->text(model->project().name)
                                                                : std::string_view(directory_name),
                                         set.sources);
        named.source_format = std::string(source_format);
        named.source_sha256 = set.digest;
        out.model_name = named.model_name;
        out.options_json = json::dump(options_document(*stage, out.targets, out.limit));
        named.compile_options = out.options_json;
        auto written = io::backend::fuzzybits::pcfb::write(passes.output.to_handle(), named);
        diagnostics.append(written.diagnostics());
        out.seconds.write = seconds_since(write_start);
        if (!written) {
            return fail(std::move(diagnostics));
        }
        out.bytes = std::move(written).value().bytes;
    } catch (...) {
        progress.finish(true);
        throw;
    }
    progress.finish(false);
    return compile_result::ok(std::move(out), std::move(diagnostics));
}

[[nodiscard]] auto has_ftp_extension(std::filesystem::path const &path) -> bool {
    return lower_case(path.extension().string()) == ".ftp";
}

[[nodiscard]] auto looks_like_ftp_text(std::filesystem::path const &path) -> bool {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    std::string head(64 * 1024, '\0');
    input.read(head.data(), static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<std::size_t>(input.gcount()));
    if (head.find('\0') != std::string::npos) {
        return false;
    }
    std::size_t cursor = head.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0;
    while (cursor < head.size() && std::isspace(static_cast<unsigned char>(head[cursor]))) {
        ++cursor;
    }
    if (cursor < head.size() && head[cursor] == '<') {
        return false;
    }
    for (std::size_t line = cursor; line < head.size();) {
        auto end = head.find('\n', line);
        if (end == std::string::npos) {
            end = head.size();
        }
        std::string_view text(head.data() + line, end - line);
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
            text.remove_prefix(1);
        }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
            text.remove_suffix(1);
        }
        if (text == "ENDTREE") {
            return true;
        }
        line = end + 1;
    }
    return false;
}

[[nodiscard]] auto is_ftp_path(std::filesystem::path const &path) -> bool {
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
        for (auto iterator = std::filesystem::directory_iterator(path, error);
             !error && iterator != std::filesystem::directory_iterator{}; iterator.increment(error)) {
            if (std::error_code status; iterator->is_regular_file(status) && has_ftp_extension(iterator->path())) {
                return true;
            }
        }
        return false;
    }
    return has_ftp_extension(path) || looks_like_ftp_text(path);
}

[[nodiscard]] auto diagnostics_json(api::diagnostic_list const &diagnostics) -> json::value {
    auto list = json::value::make_array();
    for (auto const &item : diagnostics.items()) {
        auto entry = json::value::make_object();
        entry.set("severity", json::value::make_string(item.severity == api::diagnostic_severity::error     ? "error"
                                                       : item.severity == api::diagnostic_severity::warning ? "warning"
                                                                                                            : "note"));
        entry.set("code", json::value::make_string(item.code));
        entry.set("message", json::value::make_string(item.message));
        list.push(std::move(entry));
    }
    return list;
}

[[nodiscard]] auto count(const std::size_t value) -> json::value {
    return json::value::make_int(static_cast<std::int64_t>(value));
}

} // namespace

auto classify_inputs(const std::span<std::filesystem::path const> inputs) -> input_kind {
    std::size_t ftp = 0;
    for (auto const &path : inputs) {
        ftp += is_ftp_path(path) ? 1 : 0;
    }
    if (ftp == 0) {
        return input_kind::other;
    }
    return ftp == inputs.size() ? input_kind::ftp : input_kind::mixed;
}

auto parse_override(const std::string_view text) -> api::result<override_spec> {
    const auto equals = text.rfind('=');
    if (equals == std::string_view::npos || equals == 0 ||
        (text.substr(equals + 1) != "0" && text.substr(equals + 1) != "1")) {
        return api::result<override_spec>::fail(
            option_error("override " + in_quotes(text) + " must be NAME=0 or NAME=1"));
    }
    return api::result<override_spec>::ok(
        override_spec{.name = std::string(text.substr(0, equals)), .value = text[equals + 1] == '1'});
}

auto compile_sources(const std::span<reader::source_text const> sources, options const &settings) -> compile_result {
    return compile_set(sources, settings, {});
}

auto compile_project(const std::span<std::filesystem::path const> inputs, options const &settings) -> compile_result {
    auto files = reader::project_files(inputs);
    if (!files) {
        return compile_result::fail(files.diagnostics());
    }
    std::vector<reader::source_text> sources;
    for (auto const &path : files.value()) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return compile_result::fail(
                api::make_error("io.ftp.open-failed", "failed to open FTP input file '" + path.generic_string() + "'"));
        }
        std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        sources.push_back(reader::source_text{.path = path.generic_string(), .text = std::move(text)});
    }
    std::error_code error;
    std::string name;
    if (inputs.size() == 1 && std::filesystem::is_directory(inputs.front(), error)) {
        auto directory = std::filesystem::absolute(inputs.front(), error).lexically_normal();
        if (directory.filename().empty()) {
            directory = directory.parent_path();
        }
        name = directory.filename().string();
    }
    return compile_set(sources, settings, std::move(name));
}

auto lower_sources(const std::span<reader::source_text const> sources, options const &settings)
    -> api::result<lowering_ns::pra_pc_lowering> {
    api::diagnostic_list diagnostics;
    timings seconds;
    source_set set;
    auto stage = run_lower(sources, settings, diagnostics, seconds, nullptr, set);
    if (!stage) {
        return api::result<lowering_ns::pra_pc_lowering>::fail(std::move(diagnostics));
    }
    return api::result<lowering_ns::pra_pc_lowering>::ok(std::move(stage->lowering), std::move(diagnostics));
}

auto report_json(compile_result const &result, const unsigned indent) -> std::string {
    auto root = json::value::make_object();
    root.set("status", json::value::make_string(result ? "ok" : "error"));
    if (result.has_value()) {
        auto const &compiled = result.value();
        auto source = json::value::make_object();
        source.set("format", json::value::make_string(std::string(source_format)));
        auto files = json::value::make_array();
        for (auto const &file : compiled.files) {
            files.push(json::value::make_string(file));
        }
        source.set("files", std::move(files));
        source.set("sha256", json::value::make_string(compiled.source_sha256));
        if (!compiled.limit.empty()) {
            source.set("limit", json::value::make_string(compiled.limit));
        }
        root.set("source", std::move(source));
        root.set("model_name", json::value::make_string(compiled.model_name));
        if (auto parsed = json::parse(compiled.options_json)) {
            root.set("options", std::move(parsed).value());
        }
        auto lowering = json::value::make_object();
        auto const &report = compiled.lowering;
        lowering.set("transform", json::value::make_string(report.transform));
        lowering.set("version", json::value::make_int(report.version));
        lowering.set("semantics",
                     json::value::make_string(report.semantics == lowering_ns::semantic_class::function_preserving
                                                  ? "function_preserving"
                                                  : "distribution_preserving"));
        lowering.set("source_gates", count(report.source_gates));
        lowering.set("source_operands", count(report.source_operands));
        lowering.set("source_basic_events", count(report.source_basic_events));
        lowering.set("source_house_events", count(report.source_house_events));
        lowering.set("overridden_elements", count(report.overridden_elements));
        lowering.set("pc_nodes", count(report.pc_nodes));
        lowering.set("pc_edges", count(report.pc_edges));
        lowering.set("pc_targets", count(report.pc_targets));
        lowering.set("generated_nodes", count(report.generated_nodes));
        lowering.set("generated_edges", count(report.generated_edges));
        lowering.set("conditioned_targets", count(report.conditioned_targets));
        lowering.set("conditioned_nodes", count(report.conditioned_nodes));
        root.set("lowering", std::move(lowering));
        auto passes = json::value::make_array();
        for (auto const &pass : compiled.passes) {
            auto entry = json::value::make_object();
            entry.set("transform", json::value::make_string(pass.transform));
            entry.set("semantics", json::value::make_string(pass.semantics));
            entry.set("nodes_before", json::value::make_int(pass.nodes_before));
            entry.set("edges_before", json::value::make_int(pass.edges_before));
            entry.set("nodes_after", json::value::make_int(pass.nodes_after));
            entry.set("edges_after", json::value::make_int(pass.edges_after));
            entry.set("rewrites", json::value::make_int(static_cast<std::int64_t>(pass.rewrites)));
            entry.set("skipped", json::value::make_int(static_cast<std::int64_t>(pass.skipped)));
            passes.push(std::move(entry));
        }
        root.set("passes", std::move(passes));
        auto circuit = json::value::make_object();
        circuit.set("nodes_before", json::value::make_int(compiled.nodes_before));
        circuit.set("edges_before", json::value::make_int(compiled.edges_before));
        circuit.set("nodes_after", json::value::make_int(compiled.nodes_after));
        circuit.set("edges_after", json::value::make_int(compiled.edges_after));
        circuit.set("bytes", count(compiled.bytes.size()));
        root.set("circuit", std::move(circuit));
        auto seconds = json::value::make_object();
        seconds.set("read", json::value::make_number(compiled.seconds.read));
        seconds.set("lower", json::value::make_number(compiled.seconds.lower));
        seconds.set("passes", json::value::make_number(compiled.seconds.passes));
        seconds.set("write", json::value::make_number(compiled.seconds.write));
        root.set("seconds", std::move(seconds));
    }
    root.set("diagnostics", diagnostics_json(result.diagnostics()));
    return json::dump(root, indent);
}

} // namespace fuzzybits::compile::ftp

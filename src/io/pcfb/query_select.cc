#include "io/pcfb/query_select.h"

#include "api/diagnostics.h"
#include "ir/pc/op_code.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb {

    namespace {

        [[nodiscard]] auto error(std::string code, std::string message) -> api::diagnostic {
            return api::make_error(std::move(code), std::move(message));
        }

        [[nodiscard]] auto kinds_of(std::span<query_info const> queries, const std::string_view name)
            -> std::vector<query_kind> {
            std::vector<query_kind> kinds;
            for (auto const &query : queries) {
                if (query.name == name && std::ranges::find(kinds, query.kind) == kinds.end()) {
                    kinds.push_back(query.kind);
                }
            }
            return kinds;
        }

        [[nodiscard]] auto kind_list(std::vector<query_kind> const &kinds) -> std::string {
            std::string out;
            for (std::size_t index = 0; index < kinds.size(); ++index) {
                if (index != 0) {
                    out += index + 1 == kinds.size() ? " and " : ", ";
                }
                out += query_kind_name(kinds[index]);
            }
            return out;
        }

        [[nodiscard]] auto names_of(std::span<query_info const> queries) -> std::string {
            std::vector<std::string_view> names;
            for (auto const &query : queries) {
                if (std::ranges::find(names, std::string_view{query.name}) == names.end()) {
                    names.push_back(query.name);
                }
            }
            std::string out;
            for (std::size_t index = 0; index < names.size(); ++index) {
                if (index != 0) {
                    out += ", ";
                }
                out += "'" + std::string{names[index]} + "'";
            }
            return out;
        }

        [[nodiscard]] auto select_by_kind(std::span<query_info const> queries, const query_kind kind,
                                          const std::optional<std::string_view> name) -> api::result<query_selection> {
            using selection_result = api::result<query_selection>;
            for (auto const &query : queries) {
                if (query.kind != kind || (name && query.name != *name)) {
                    continue;
                }
                return selection_result::ok(query_selection{
                    .index = query.index,
                    .target = query.target,
                    .kind = query.kind,
                    .name = query.name,
                });
            }
            return selection_result::fail(error(
                "io.pcfb.query-not-found",
                name ? "no query named '" + std::string(*name) + "' has kind " + std::string(query_kind_name(kind))
                     : "the file has no query of kind " + std::string(query_kind_name(kind))));
        }

    } // namespace

    auto query_kind_name(const query_kind kind) noexcept -> std::string_view {
        switch (kind) {
        case query_kind::probability:
            return "probability";
        case query_kind::monotone_hull_probability:
            return "hull";
        }
        return "probability";
    }

    auto parse_query_kind(const std::string_view text) -> std::optional<query_kind> {
        if (text == "probability") {
            return query_kind::probability;
        }
        if (text == "hull") {
            return query_kind::monotone_hull_probability;
        }
        return std::nullopt;
    }

    auto queries_of(read_output const &file) -> std::vector<query_info> {
        std::vector<query_info> queries;
        queries.reserve(file.queries.size());
        for (std::uint32_t index = 0; index < file.queries.size(); ++index) {
            const auto target = file.queries[index].target;
            const auto named = file.options.target_names.find(target);
            queries.push_back(query_info{
                .index = index,
                .target = target,
                .kind = file.queries[index].kind,
                .name = named == file.options.target_names.end() ? "t" + std::to_string(target.raw()) : named->second,
            });
        }
        return queries;
    }

    auto select_query(std::span<query_info const> queries, const std::optional<std::string_view> name,
                      const std::optional<query_kind> kind) -> api::result<query_selection> {
        using selection_result = api::result<query_selection>;
        if (queries.empty()) {
            return selection_result::fail(error("io.pcfb.query-not-found", "the file has no queries"));
        }
        if (!name) {
            if (kind) {
                return select_by_kind(queries, *kind, std::nullopt);
            }
            auto const &first = queries.front();
            return selection_result::ok(query_selection{
                .index = first.index,
                .target = first.target,
                .kind = first.kind,
                .name = first.name,
            });
        }
        const auto kinds = kinds_of(queries, *name);
        if (kinds.empty()) {
            return selection_result::fail(error("io.pcfb.query-not-found",
                                                "no query named '" + std::string(*name) + "'; the file queries " +
                                                    names_of(queries)));
        }
        if (kind) {
            if (std::ranges::find(kinds, *kind) == kinds.end()) {
                return selection_result::fail(error("io.pcfb.query-not-found",
                                                    "query '" + std::string(*name) + "' has kind " + kind_list(kinds) +
                                                        ", not " + std::string(query_kind_name(*kind))));
            }
            return select_by_kind(queries, *kind, name);
        }
        if (kinds.size() > 1) {
            return selection_result::fail(error("io.pcfb.query-ambiguous",
                                                "query '" + std::string(*name) + "' is written as " + kind_list(kinds) +
                                                    "; select one with --kind probability|hull"));
        }
        return select_by_kind(queries, kinds.front(), name);
    }

    auto select_hull_query(std::span<query_info const> queries, const std::optional<std::string_view> name,
                           const std::optional<query_kind> kind) -> api::result<query_selection> {
        using selection_result = api::result<query_selection>;
        if (!name && !kind) {
            const auto hulls = static_cast<std::size_t>(std::ranges::count_if(
                queries, [](query_info const &query) { return query.kind == query_kind::monotone_hull_probability; }));
            if (hulls == 1) {
                for (auto const &query : queries) {
                    if (query.kind == query_kind::monotone_hull_probability) {
                        return selection_result::ok(query_selection{
                            .index = query.index,
                            .target = query.target,
                            .kind = query.kind,
                            .name = query.name,
                        });
                    }
                }
            }
            if (hulls > 1) {
                std::string listed;
                for (auto const &query : queries) {
                    if (query.kind != query_kind::monotone_hull_probability) {
                        continue;
                    }
                    if (!listed.empty()) {
                        listed += ", ";
                    }
                    listed += "'" + query.name + "' (query " + std::to_string(query.index) + ")";
                }
                return selection_result::fail(
                    error("io.pcfb.query-ambiguous", "the file has " + std::to_string(hulls) + " hull queries (" +
                                                         listed + "); select one with --query NAME"));
            }
        }
        auto selected = select_query(queries, name, kind);
        if (!selected) {
            return selected;
        }
        if (selected.value().kind != query_kind::monotone_hull_probability) {
            return selection_result::fail(error(
                "io.pcfb.query-not-hull",
                "query " + std::to_string(selected.value().index) + " ('" + selected.value().name + "') has kind " +
                    std::string(query_kind_name(selected.value().kind)) +
                    ", not hull; select a hull query with --query NAME or --kind hull"));
        }
        return selected;
    }

    auto basic_event_named(read_output const &file, const std::string_view name)
        -> api::result<ir::pc::bool_node_id> {
        using node_result = api::result<ir::pc::bool_node_id>;
        for (std::uint32_t node = 0; node < file.circuit.node_count(); ++node) {
            const auto id = ir::pc::bool_node_id(node);
            const auto named = file.options.node_names.find(id);
            if (named == file.options.node_names.end() || named->second != name) {
                continue;
            }
            if (file.circuit.op_codes()[node] != ir::pc::op_code::basic_event) {
                return node_result::fail(
                    error("io.pcfb.not-an-event", "'" + std::string(name) + "' is not a basic event"));
            }
            return node_result::ok(id);
        }
        return node_result::fail(error("io.pcfb.unknown-event", "no basic event named '" + std::string(name) + "'"));
    }

    auto node_name(read_output const &file, const ir::pc::bool_node_id node) -> std::string {
        const auto named = file.options.node_names.find(node);
        return named == file.options.node_names.end() ? "e" + std::to_string(node.raw()) : named->second;
    }

} // namespace fuzzybits::io::backend::fuzzybits::pcfb

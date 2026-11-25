"""Python ``fzb`` console script. Flag surface matches ``targets/fzb/main.cpp``."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from . import __version__, _runtime
from ._api import compile as compile_pcfb
from ._api import GroupResult, _ftp_inspect_native, exact, exact_groups, inspect, pcfb_to_nombrilo, run

ROOT_USAGE = (
    "Usage: fzb <command> [options]\n\n"
    "Commands:\n"
    "  compile   OpenPSA/MEF XML, nombrilo.circuit or CAFTA FTP projects → .pcfb\n"
    "  export    .pcfb → nombrilo.circuit (.safetensors)\n"
    "  inspect   Summarise, validate, or exactly evaluate a .pcfb\n"
    "  run       Monte Carlo a .pcfb (estimate and CI per query)\n"
    "  serve     Start fuzzybits-server (API and web UI); fzb serve --help lists its options\n"
)

SERVER_DISTRIBUTION = "fuzzybits-server"
SERVE_MISSING = "fzb serve needs the fuzzybits-server package: pip install 'fuzzybits[server]'\n"

INSPECT_JSON_KEYS = (
    "schema_version",
    "tensors",
    "operators",
    "samples",
    "queries",
    "opcodes",
    "sample_source",
    "query_names",
    "query_kinds",
    "metadata",
)

SEED_REJECTED = (
    "fzb run: --seed is not supported yet; results are deterministic "
    "per model and trial count (Philox counters are "
    "{pdag_idx+1, event_idx+1, sample_idx+1, (iteration+1)<<6}, "
    "not Settings::seed). The only accepted value is 372, the "
    "scram-cli default.\n"
)

SCRAM_CLI_DEFAULT_SEED = 372


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - argparse hook
        self.print_help(sys.stderr)
        print(f"Option error: {message}", file=sys.stderr)
        raise SystemExit(1)


def _print_root_usage(out: TextIO) -> None:
    out.write(ROOT_USAGE)


def _cli_error(command: str, exc: BaseException) -> int:
    msg = str(exc)
    for prefix in (f"{command}: ", "compile: ", "export: ", "inspect: ", "run: "):
        if msg.startswith(prefix):
            msg = msg[len(prefix) :]
            break
    print(f"fzb {command}: {msg}", file=sys.stderr)
    return 1


def _add_progress_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared with `targets/fzb/main.cpp`; the C++ layer owns the decision."""
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="do not draw the progress line (it is on by default on a terminal)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="no progress line and no informational stderr; results and errors still print",
    )


def _progress_enabled(args: argparse.Namespace) -> bool | None:
    """Map the flags onto the native ``progress=`` argument.

    ``--no-progress`` / ``--quiet`` are passed as ``False`` so they win over the environment;
    otherwise ``None`` leaves the automatic C++ policy (stderr TTY, ``TERM``,
    ``FUZZYBITS_PROGRESS``) in charge.
    """
    if args.no_progress or args.quiet:
        return False
    return None


def _parse_trials(text: str) -> int:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid --trials '{text}'") from exc
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"invalid --trials '{text}'")
    return int(value)


def _expand_name_args(values: Sequence[str]) -> list[str]:
    """Replace each ``@FILE`` value by the names in FILE, one per line; never split on commas."""
    names: list[str] = []
    for value in values:
        if not value.startswith("@"):
            names.append(value)
            continue
        path = value[1:]
        if os.path.exists(path) and not os.path.isfile(path):
            raise ValueError(f"{path} is not a regular file")
        try:
            with open(path, encoding="utf-8", newline="") as handle:
                lines = handle.read().split("\n")
        except (OSError, UnicodeDecodeError):
            raise ValueError(f"cannot read {path}") from None
        found = [line.strip() for line in lines if line.strip()]
        if not found:
            raise ValueError(f"{path} lists no names")
        names.extend(found)
    return names


def _compile_ftp_command(args: argparse.Namespace, flags: Mapping[str, Any]) -> int:
    """The FTP project path of ``fzb compile``; mirrors ``compile_ftp_inputs`` in ``targets/fzb/main.cpp``."""
    for flag in (
        "no_kn",
        "no_xor",
        "recognize_atleast",
        "no_recognize_atleast",
        "recognize_xor",
        "no_recognize_xor",
        "keep_null",
        "allow_extern",
        "weights",
        "no_ccf",
    ):
        if getattr(args, flag):
            print(
                f"fzb compile: --{flag.replace('_', '-')} applies to OpenPSA/MEF input, not to FTP projects",
                file=sys.stderr,
            )
            return 1
    overrides: dict[str, bool] = {}
    for text in args.override:
        name, equals, value = text.rpartition("=")
        if not equals or not name or value not in ("0", "1"):
            print(f"fzb compile: override '{text}' must be NAME=0 or NAME=1", file=sys.stderr)
            return 1
        overrides[name] = value == "1"
    options: dict[str, Any] = {}
    if args.target:
        options["target"] = list(args.target)
    if args.watch:
        options["watch"] = list(args.watch)
    if overrides:
        options["overrides"] = overrides
    if args.kind:
        options["kind"] = list(args.kind)
    for key in ("duplicate_import", "hull_literals", "initiator_values"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if args.hull_max_initiators is not None:
        text = args.hull_max_initiators
        if text == "none":
            options["hull_max_initiators"] = None
        else:
            try:
                count = int(text)
            except ValueError:
                count = -1
            if count < 0 or not text.isdigit():
                print(
                    f"fzb compile: --hull-max-initiators must be a non-negative integer or none, not '{text}'",
                    file=sys.stderr,
                )
                return 1
            options["hull_max_initiators"] = count
    if args.config is not None:
        try:
            options["config"] = Path(args.config).read_text(encoding="utf-8")
        except OSError:
            print(f"fzb compile: cannot read {args.config}", file=sys.stderr)
            return 1
    try:
        blob, document = compile_pcfb(
            args.input_file,
            pdag_c=args.compilation_passes,
            report=True,
            progress=_progress_enabled(args),
            progress_log=args.progress_log,
            **options,
        )
    except Exception as exc:
        return _cli_error("compile", exc)

    meta = inspect(blob).get("metadata") or {}
    model = str(meta.get("general.model_name") or "")
    out_path = Path(args.output) if args.output else Path(Path(model).name + ".pcfb")
    try:
        out_path.write_bytes(blob)
        if args.report:
            Path(args.report).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"fzb compile: cannot write {exc.filename or out_path}", file=sys.stderr)
        print(f"fzb compile: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        circuit = document["circuit"]
        lowering = document["lowering"]
        diagnostics = document["diagnostics"]
        targets = [item["name"] for item in document["options"]["targets"] + document["options"]["watch"]]
        print(
            f"wrote {out_path} ({len(blob)} bytes) model={model} targets={len(targets)} ({','.join(targets)})\n"
            f"  source: {len(document['source']['files'])} file(s) sha256={document['source']['sha256'][:16]}"
            f" c{document['options']['compilation_level']}\n"
            f"  lowered: {lowering['source_gates']} gates, {lowering['source_basic_events']} basic events, "
            f"{lowering['source_house_events']} house events, {lowering['overridden_elements']} overrides\n"
            f"  circuit: nodes {circuit['nodes_before']} -> {circuit['nodes_after']}, "
            f"edges {circuit['edges_before']} -> {circuit['edges_after']}\n"
            f"  diagnostics: {sum(d['severity'] == 'warning' for d in diagnostics)} warning(s), "
            f"{sum(d['severity'] == 'note' for d in diagnostics)} note(s)",
            file=sys.stderr,
        )
    return 0


def cmd_compile(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb compile",
        usage="fzb compile model.xml|model.safetensors [-c N] [--no-kn] [--no-ccf] "
        "[--target NAME|@FILE] [--watch NAME|@FILE]... [--weights FILE] [-o model.pcfb]\n"
        "       fzb compile a.ftp [b.ftp ...]|DIR [-c 0|1] [--target NAME|@FILE]... [--watch NAME|@FILE]... "
        "[--kind probability|hull]... [--hull-literals split|fixed|plain] [--hull-max-initiators N|none] "
        "[--initiator-values UNIT|probability] [--override NAME=0|1]... [--config FILE] [--report FILE] "
        "[-o model.pcfb]\n"
        "                       [--no-progress] [--quiet] [--progress-log FILE]",
        description="fzb compile options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("-o", "--output", help="output .pcfb path")
    parser.add_argument(
        "-c",
        "--compilation-passes",
        type=int,
        default=None,
        help="pass level. MEF/nombrilo (SCRAM preprocessing): 0=off 1=null-only 2=optimize (default) 4-8=multipass. "
        "FTP (ir::pc passes): 0=none 1=all (default)",
    )
    parser.add_argument("--no-kn", action="store_true", help="expand k/n to and/or")
    parser.add_argument("--no-xor", action="store_true", help="expand xor to and/or")
    parser.add_argument(
        "--recognize-atleast",
        action="store_true",
        help="recognize expanded threshold groups (default on)",
    )
    parser.add_argument(
        "--no-recognize-atleast",
        action="store_true",
        help="disable k/n group recognition",
    )
    parser.add_argument("--recognize-xor", action="store_true", help="recognize expanded xor (default on)")
    parser.add_argument("--no-recognize-xor", action="store_true", help="disable xor group recognition")
    parser.add_argument("--keep-null", action="store_true", help="keep NULL/pass-through gates")
    parser.add_argument(
        "--no-ccf",
        action="store_true",
        help="MEF: do not expand common-cause failure groups (expanded by default)",
    )
    parser.add_argument("--allow-extern", action="store_true", help="**UNSAFE** allow external libraries")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="MEF: top-event / gate / nombrilo name (one). FTP: PROCESS target, gate or basic event "
        "(repeatable; default: every PROCESS target). @FILE reads one name per line",
    )
    parser.add_argument("--override", action="append", default=[], help="FTP: force an element, NAME=0 or NAME=1 (repeatable)")
    parser.add_argument("--duplicate-import", help="FTP: event imported more than once: error|first|last (default last)")
    parser.add_argument(
        "--kind",
        action="append",
        default=[],
        help="FTP: the query of every target, probability (default) or hull; repeat it, or give probability,hull, "
        "for both (two queries per target)",
    )
    parser.add_argument(
        "--hull-literals",
        help="FTP hull queries: how -EVENT operands are read, split (an atom of its own, default), fixed "
        "(Rauzy's omega_L) or plain (an ordinary complement)",
    )
    parser.add_argument(
        "--hull-max-initiators",
        help="FTP hull queries, with --initiator-values probability: delete the cut sets with more than N "
        "initiating events, or none (default)",
    )
    parser.add_argument(
        "--initiator-values",
        help="FTP: what I values are: a frequency in per-year (default), per-month, per-week, per-day or per-hour "
        "(one conditioned query per initiating event, summed as the target's frequency), or probability "
        "(values up to 1)",
    )
    parser.add_argument("--config", metavar="FILE", help="FTP: quantification config FILE (JSON); flags override it")
    parser.add_argument(
        "--report",
        metavar="FILE",
        help="FTP: write the lowering and pass reports and diagnostics to FILE (JSON)",
    )
    parser.add_argument("--weights", help="nombrilo variant weights file")
    parser.add_argument(
        "--progress-log",
        metavar="FILE",
        help="write a progress log to FILE (.jsonl or .csv; all snapshot fields,"
        " independent of the terminal display)",
    )
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        help="extra gate / basic event written as an additional query (repeatable; NAME or @FILE)",
    )
    _add_progress_flags(parser)
    parser.add_argument(
        "input_file", nargs="*", help="MEF XML, nombrilo.circuit, or CAFTA FTP files / a directory of *.ftp files"
    )
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb compile: missing input file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    try:
        args.target = _expand_name_args(args.target)
        args.watch = _expand_name_args(args.watch)
    except ValueError as exc:
        print(f"fzb compile: {exc}", file=sys.stderr)
        return 1

    try:
        kind = _runtime.native().classify_compile_inputs(list(args.input_file))
    except Exception as exc:
        return _cli_error("compile", exc)
    if kind == "mixed":
        print("fzb compile: FTP and non-FTP inputs cannot be combined", file=sys.stderr)
        return 1
    ftp_flags = {
        "override": args.override,
        "duplicate-import": args.duplicate_import,
        "kind": args.kind,
        "hull-literals": args.hull_literals,
        "hull-max-initiators": args.hull_max_initiators,
        "initiator-values": args.initiator_values,
        "config": args.config,
        "report": args.report,
    }
    if kind == "ftp":
        return _compile_ftp_command(args, ftp_flags)
    for flag, value in ftp_flags.items():
        if value:
            print(f"fzb compile: --{flag} applies to CAFTA FTP input only", file=sys.stderr)
            return 1
    seen: set[str] = set()
    for name in [*args.target, *args.watch]:
        if name in seen:
            print(f"fzb compile: target '{name}' is requested more than once", file=sys.stderr)
            return 1
        seen.add(name)
    if len(args.target) > 1:
        print(
            "fzb compile: MEF and nombrilo inputs take one --target; several targets need the new MEF reader (#148)",
            file=sys.stderr,
        )
        return 1

    options: dict[str, Any] = {
        "no_xor": bool(args.no_xor),
        "keep_null": bool(args.keep_null),
        "ccf": not args.no_ccf,
        "allow_extern": bool(args.allow_extern),
    }
    if args.no_recognize_atleast:
        options["recognize_atleast_groups"] = False
    elif args.recognize_atleast:
        options["recognize_atleast_groups"] = True
    if args.no_recognize_xor:
        options["recognize_xor_groups"] = False
    elif args.recognize_xor:
        options["recognize_xor_groups"] = True
    if args.target:
        options["target"] = args.target[0]
    if args.weights:
        options["weights"] = args.weights
    if args.watch:
        options["watch"] = list(args.watch)

    try:
        blob = compile_pcfb(
            args.input_file,
            pdag_c=2 if args.compilation_passes is None else int(args.compilation_passes),
            no_kn=bool(args.no_kn),
            progress=_progress_enabled(args),
            progress_log=args.progress_log,
            **options,
        )
    except Exception as exc:
        return _cli_error("compile", exc)

    if args.output:
        out_path = Path(args.output)
    else:
        # Native default is `{stem}.pcfb` in the process cwd, not beside the XML.
        out_path = Path(Path(args.input_file[0]).stem + ".pcfb")
    try:
        out_path.write_bytes(blob)
    except OSError as exc:
        print(f"fzb compile: cannot write {out_path}", file=sys.stderr)
        print(f"fzb compile: {exc}", file=sys.stderr)
        return 1

    model = ""
    top = ""
    try:
        meta = inspect(blob).get("metadata") or {}
        model = str(meta.get("general.model_name") or "")
        top = str(meta.get("fuzzybits.target") or "")
    except Exception:
        pass
    if not args.quiet:
        print(
            f"wrote {out_path} ({len(blob)} bytes) model={model} top={top}",
            file=sys.stderr,
        )
    return 0


def cmd_export(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb export",
        usage="fzb export --to nombrilo model.pcfb -o out.safetensors [--strict]",
        description="fzb export options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("--to", help="target format (nombrilo)")
    parser.add_argument("-o", "--output", help="output path")
    parser.add_argument("--strict", action="store_true", help="omit fuzzybits.pcfb.* extras")
    parser.add_argument("input_file", nargs="?", help=".pcfb file")
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb export: missing .pcfb file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    if args.to not in {"nombrilo", "nombrilo.circuit"}:
        print("fzb export: --to nombrilo is required\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    try:
        if args.strict and _runtime.native().query_groups_pcfb(Path(args.input_file).read_bytes()):
            print(
                "fzb export: warning: --strict drops the file's query groups (frequencies); without --strict "
                "they are kept in fuzzybits.pcfb.query_groups",
                file=sys.stderr,
            )
        blob = pcfb_to_nombrilo(args.input_file, extras=not args.strict)
    except Exception as exc:
        return _cli_error("export", exc)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(Path(args.input_file).stem + ".safetensors")
    try:
        out_path.write_bytes(blob)
    except OSError as exc:
        print(f"fzb export: cannot write {out_path}", file=sys.stderr)
        print(f"fzb export: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out_path} ({len(blob)} bytes)", file=sys.stderr)
    return 0


def _inspect_text(summary: Mapping[str, Any], out: TextIO) -> None:
    schema = int(summary.get("schema_version") or 0)
    out.write(f"schema_version  0x{schema:x}\n")
    out.write(f"tensors         {int(summary.get('tensors') or 0)}\n")
    out.write(f"operators       {int(summary.get('operators') or 0)}\n")
    out.write(f"samples         {int(summary.get('samples') or 0)}\n")
    out.write(f"queries         {int(summary.get('queries') or 0)}\n")
    opcodes = summary.get("opcodes") or {}
    if opcodes:
        out.write("opcodes\n")
        for name, n in opcodes.items():
            out.write(f"  {name}  {n}\n")
    sources = summary.get("sample_source") or {}
    if sources:
        out.write("sample_source\n")
        for name, n in sources.items():
            out.write(f"  {name}  {n}\n")
    names = summary.get("query_names") or []
    if names:
        out.write("query_names\n")
        for name in names:
            out.write(f"  {name}\n")
        out.write("query_kinds\n")
        for kind in summary.get("query_kinds") or []:
            out.write(f"  {kind}\n")
    metadata = summary.get("metadata") or {}
    if metadata:
        out.write("metadata\n")
        for key, value in metadata.items():
            out.write(f"  {key}  {value}\n")


def _inspect_json(summary: Mapping[str, Any], out: TextIO) -> None:
    payload = {key: summary.get(key) for key in INSPECT_JSON_KEYS}
    json.dump(payload, out, ensure_ascii=False, indent=2)
    out.write("\n")


def _inspect_ftp_command(args: argparse.Namespace) -> int:
    for flag in ("validate", "exact"):
        if getattr(args, flag):
            print(f"fzb inspect: --{flag} applies to .pcfb input only", file=sys.stderr)
            return 1
    try:
        document, text = _ftp_inspect_native(
            list(args.input_file),
            args.duplicate_import or "last",
            args.counts_only,
        )
    except Exception as exc:
        return _cli_error("inspect", exc)
    if args.json:
        sys.stdout.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(text)
    severities = [item["severity"] for item in document["diagnostics"]]
    if "error" in severities or (args.strict and "warning" in severities):
        return 1
    return 0


def cmd_inspect(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb inspect",
        usage=(
            "fzb inspect model.pcfb [--validate] [--json] [--exact [--exposure DURATION]]\n"
            "       fzb inspect a.ftp [b.ftp ...]|DIR [--json] [--duplicate-import error|first|last] "
            "[--strict] [--counts-only]"
        ),
        description="fzb inspect options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("--validate", action="store_true", help="run pcfb::validate; exit non-zero on diagnostics")
    parser.add_argument("--json", action="store_true", help="print summary as JSON")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="exhaustive exact P(query) for small models (≤ 30 events), and the exact value of every frequency group",
    )
    parser.add_argument("--exposure", metavar="DURATION", help="with --exact: as for fzb run")
    parser.add_argument("--duplicate-import", help="FTP: event imported more than once: error|first|last (default last)")
    parser.add_argument("--strict", action="store_true", help="FTP: exit 1 on warnings as well as errors")
    parser.add_argument("--counts-only", action="store_true", help="FTP: print counts without element names")
    parser.add_argument(
        "input_file", nargs="*", help=".pcfb file, or CAFTA FTP files / a directory of *.ftp files"
    )
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb inspect: missing .pcfb file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    try:
        kind = _runtime.native().classify_compile_inputs(list(args.input_file))
    except Exception as exc:
        return _cli_error("inspect", exc)
    if kind == "mixed":
        print("fzb inspect: FTP and non-FTP inputs cannot be combined", file=sys.stderr)
        return 1
    if kind == "ftp":
        return _inspect_ftp_command(args)
    if kind == "ftp" and args.exposure:
        print("fzb inspect: --exposure applies to .pcfb input only", file=sys.stderr)
        return 1
    if args.exposure and not args.exact:
        print("fzb inspect: --exposure needs --exact", file=sys.stderr)
        return 1
    for flag, value in (
        ("duplicate-import", args.duplicate_import),
        ("strict", args.strict),
        ("counts-only", args.counts_only),
    ):
        if value:
            print(f"fzb inspect: --{flag} applies to CAFTA FTP input only", file=sys.stderr)
            return 1
    if len(args.input_file) > 1:
        parser.error("too many positional options have been specified on the command line")
    args.input_file = args.input_file[0]

    try:
        summary = inspect(args.input_file)
    except Exception as exc:
        return _cli_error("inspect", exc)

    rc = 0
    if args.validate:
        diagnostics = list(summary.get("diagnostics") or [])
        if diagnostics:
            rc = 1
            for diag in diagnostics:
                if isinstance(diag, Mapping):
                    location = diag.get("location", "")
                    message = diag.get("message", diag)
                else:
                    location = getattr(diag, "location", "")
                    message = getattr(diag, "message", diag)
                print(f"validate: {location}: {message}", file=sys.stderr)
        elif not args.json:
            print("validate: ok", file=sys.stderr)

    if args.json:
        _inspect_json(summary, sys.stdout)
    else:
        _inspect_text(summary, sys.stdout)

    if args.exact:
        try:
            p = exact(args.input_file)
        except Exception as exc:
            msg = str(exc)
            print(f"fzb inspect --exact: {msg}", file=sys.stderr)
            return 1
        print(f"exact_probability  {p:.16e}")
        try:
            groups = exact_groups(args.input_file, exposure=args.exposure)
        except Exception as exc:
            print(f"fzb inspect --exact: {exc}", file=sys.stderr)
            return 1
        _print_groups(groups, args.exposure, "exact_group  ")
    return rc


def cmd_run(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb run",
        usage="fzb run model.pcfb --trials N [--seed S] [--backend auto|cuda|hip|omp|metal] [--json] "
        "[--exposure DURATION]\n"
        "              [--no-progress] [--quiet] [--progress-log FILE]",
        description="fzb run options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("-N", "--trials", help="Bernoulli trials (e.g. 1e9, 1000000)")
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=None,
        help="must be 372 (the scram-cli default); other values are rejected",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="AdaptiveCpp backend: auto, cuda, hip, omp, or metal",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument(
        "--exposure",
        metavar="DURATION",
        help="for each frequency group, also print P(at least one occurrence in DURATION) = 1 - exp(-F * DURATION); "
        "DURATION is a number and y|mo|wk|d|h, e.g. 1y, 1mo",
    )
    parser.add_argument(
        "--progress-log",
        metavar="FILE",
        help="write a progress log to FILE (.jsonl or .csv; all snapshot fields,"
        " independent of the terminal display)",
    )
    _add_progress_flags(parser)
    parser.add_argument("input_file", nargs="?", help=".pcfb file")
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb run: missing .pcfb file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    if args.trials is None:
        print("fzb run: --trials is required\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    try:
        trials = _parse_trials(args.trials)
    except ValueError as exc:
        print(f"fzb run: {exc}", file=sys.stderr)
        return 1
    if args.seed is not None and args.seed != SCRAM_CLI_DEFAULT_SEED:
        sys.stderr.write(SEED_REJECTED)
        return 1

    backend = str(args.backend or "auto").lower()
    if backend not in {"auto", "cuda", "hip", "omp", "metal"}:
        print(
            f"fzb run: unknown backend '{args.backend}' (expected auto, cuda, hip, omp, or metal)",
            file=sys.stderr,
        )
        return 1

    try:
        rows = run(
            args.input_file,
            trials=trials,
            backend=backend,
            progress=_progress_enabled(args),
            progress_log=args.progress_log,
            exposure=args.exposure,
        )
    except Exception as exc:
        return _cli_error("run", exc)

    if args.json:
        payload = {
            "queries": [
                {
                    "name": row.name,
                    "estimate": row.estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "n": row.n,
                    "hits": row.hits,
                }
                for row in rows
            ]
        }
        if rows.groups:
            payload["groups"] = [_group_json(group, args.exposure) for group in rows.groups]
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for row in rows:
            print(
                f"{row.name}  {row.estimate:.8e}  [{row.ci_low:.8e}, {row.ci_high:.8e}]  "
                f"n={row.n}  hits={row.hits}"
            )
        _print_groups(rows.groups, args.exposure, "")
    return 0


def _server_version() -> str | None:
    """Installed ``fuzzybits-server`` distribution version, or ``None`` without metadata.

    A source checkout on ``PYTHONPATH`` has no distribution metadata; it is not warned about.
    """
    from importlib import metadata

    try:
        return metadata.version(SERVER_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def cmd_serve(argv: Sequence[str]) -> int:
    """``fzb serve``: hand the remaining arguments to ``fuzzybits_server.app.main``.

    The server's parser is the only one: ``fzb serve --help`` is its help, and its usage
    errors exit as argparse does (status 2), not with fzb's status 1.
    """
    import importlib
    import importlib.util

    if importlib.util.find_spec("fuzzybits_server") is None:
        sys.stderr.write(SERVE_MISSING)
        return 1
    server_version = _server_version()
    if server_version is not None and server_version != __version__:
        print(
            f"fzb serve: warning: fuzzybits-server {server_version} does not match fuzzybits {__version__}; "
            f"install matching versions: pip install 'fuzzybits[server]=={__version__}'",
            file=sys.stderr,
        )
    try:
        # Lazy: importing the server pulls in FastAPI and uvicorn, and builds the app. Import the
        # module by name: the package re-exports the FastAPI instance as `fuzzybits_server.app`.
        server_app = importlib.import_module("fuzzybits_server.app")
    except ImportError as exc:
        print(f"fzb serve: cannot import fuzzybits_server: {exc}", file=sys.stderr)
        sys.stderr.write(SERVE_MISSING)
        return 1
    import inspect

    if "argv" in inspect.signature(server_app.main).parameters:
        server_app.main(list(argv), prog="fzb serve")
    else:
        # A fuzzybits-server older than #203 parses sys.argv itself (the version warning above
        # has fired): give it the arguments that way.
        saved = sys.argv
        sys.argv = ["fzb serve", *argv]
        try:
            server_app.main()
        finally:
            sys.argv = saved
    return 0


def _utf8_stdio() -> None:
    """Write UTF-8 to stdout/stderr whatever the locale says.

    Usage text and ``--json`` output contain non-ASCII (``→``, ``≤``, model names). Windows
    pipes and redirects default to the ANSI code page (cp1252), which cannot encode them, so
    ``fzb --help | more`` raised UnicodeEncodeError. Python 3.15 makes UTF-8 the default
    (PEP 686); this does the same for 3.12-3.14. Interactive Windows consoles already report
    UTF-8 and are left alone. The stream's error handler is kept.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if reconfigure is None or encoding == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors=getattr(stream, "errors", None) or "strict")
        except (OSError, ValueError):
            pass


def _unit_label(unit: str) -> str:
    for spec in _runtime.native().frequency_units():
        if spec["unit"] == unit:
            return str(spec["label"])
    return unit or "(no unit)"


def _print_groups(groups: Sequence[GroupResult], exposure: str | None, prefix: str) -> None:
    """Group lines as the native ``fzb`` prints them."""
    for group in groups:
        print(
            f"{prefix}{group.name}  {group.estimate:.8e}  [{group.ci_low:.8e}, {group.ci_high:.8e}]  "
            f"{_unit_label(group.unit)}  ({len(group.terms)} conditioned queries)"
        )
        if group.exposure is not None:
            p, low, high = group.exposure
            print(f"{prefix}{group.name}  P(>=1 in {exposure})  {p:.8e}  [{low:.8e}, {high:.8e}]")


def _group_json(group: GroupResult, exposure: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": group.name,
        "unit": group.unit,
        "terms": len(group.terms),
        "estimate": group.estimate,
        "ci_low": group.ci_low,
        "ci_high": group.ci_high,
    }
    if group.exposure is not None:
        p, low, high = group.exposure
        out["exposure"] = {"duration": exposure, "probability": p, "ci_low": low, "ci_high": high}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdio()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_root_usage(sys.stderr)
        return 1
    cmd = args[0]
    if cmd in {"-h", "--help", "help"}:
        _print_root_usage(sys.stdout)
        return 0
    if cmd in {"-v", "--version", "version"}:
        print(f"fzb {__version__}")
        return 0
    try:
        if cmd == "compile":
            return cmd_compile(args[1:])
        if cmd == "export":
            return cmd_export(args[1:])
        if cmd == "inspect":
            return cmd_inspect(args[1:])
        if cmd == "run":
            return cmd_run(args[1:])
        if cmd == "serve":
            return cmd_serve(args[1:])
    except KeyboardInterrupt:
        # The native renderer already erased its block and printed the partial summary; a
        # traceback here would bury it. 128 + SIGINT is what a shell reports for Ctrl-C.
        sys.stderr.write("\n")
        sys.stderr.flush()
        return 128 + signal.SIGINT
    print(f"fzb: unknown command '{cmd}'", file=sys.stderr)
    _print_root_usage(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

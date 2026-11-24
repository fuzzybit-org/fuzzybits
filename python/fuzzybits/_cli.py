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
from ._api import _ftp_inspect_native, exact, inspect, pcfb_to_nombrilo, run

ROOT_USAGE = (
    "Usage: fzb <command> [options]\n\n"
    "Commands:\n"
    "  compile   OpenPSA/MEF XML, nombrilo.circuit or CAFTA FTP projects → .pcfb\n"
    "  export    .pcfb → nombrilo.circuit (.safetensors)\n"
    "  inspect   Summarise, validate, or exactly evaluate a .pcfb\n"
    "  run       Monte Carlo a .pcfb (estimate and CI per query)\n"
)

INSPECT_JSON_KEYS = (
    "schema_version",
    "tensors",
    "operators",
    "samples",
    "queries",
    "opcodes",
    "sample_source",
    "query_names",
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
    for key in ("duplicate_import", "complement_events", "negation", "literals"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if args.max_initiators_per_cutset is not None:
        text = args.max_initiators_per_cutset
        if text == "none":
            options["max_initiators_per_cutset"] = None
        else:
            try:
                count = int(text)
            except ValueError:
                count = -1
            if count < 0 or not text.isdigit():
                print(
                    f"fzb compile: --max-initiators-per-cutset must be a non-negative integer or none, not '{text}'",
                    file=sys.stderr,
                )
                return 1
            options["max_initiators_per_cutset"] = count
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
        "       fzb compile a.ftp [b.ftp ...]|DIR [--target NAME|@FILE]... [--watch NAME|@FILE]... [--no-ccf] "
        "[--negation boolean|hull|both] [--override NAME=0|1]... [--config FILE] [--report FILE] [-o model.pcfb]\n"
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
        help="MEF: 0=off 1=null-only 2=optimize 4-8=multipass (default 2). "
        "FTP: 0=none 1=ir::pc passes (default 1)",
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
        "--ccf",
        action="store_true",
        help="deprecated no-op: common-cause groups are expanded by default",
    )
    parser.add_argument(
        "--no-ccf",
        action="store_true",
        help="do not expand common-cause failure groups (MEF; accepted and ignored for FTP)",
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
    parser.add_argument("--complement-events", help="FTP: -EVENT operands: atom|negation (default atom: marked literals)")
    parser.add_argument(
        "--negation",
        help="FTP: query type of every target: boolean (the probability, default), hull or both",
    )
    parser.add_argument(
        "--max-initiators-per-cutset",
        help="FTP: at most N initiating events per cut set, or none (default: the model's ftp.max_initiators_per_cutset)",
    )
    parser.add_argument("--literals", help="FTP hull targets: atom_split|omega_l|negation (default atom_split)")
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

    if args.ccf and args.no_ccf:
        print("fzb compile: --ccf and --no-ccf cannot be combined", file=sys.stderr)
        return 1
    if args.ccf:
        print(
            "fzb compile: --ccf is deprecated: common-cause groups are expanded by default; use --no-ccf to disable",
            file=sys.stderr,
        )
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
        "complement-events": args.complement_events,
        "negation": args.negation,
        "max-initiators-per-cutset": args.max_initiators_per_cutset,
        "literals": args.literals,
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
            args.complement_events or "atom",
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
            "fzb inspect model.pcfb [--validate] [--json] [--exact]\n"
            "       fzb inspect a.ftp [b.ftp ...]|DIR [--json] [--duplicate-import error|first|last] "
            "[--complement-events atom|negation] [--strict] [--counts-only]"
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
        help="exhaustive exact P(query) for small models (≤ 30 events)",
    )
    parser.add_argument("--duplicate-import", help="FTP: event imported more than once: error|first|last (default last)")
    parser.add_argument("--complement-events", help="FTP: -EVENT operands: atom|negation (default atom: marked literals)")
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
    for flag, value in (
        ("duplicate-import", args.duplicate_import),
        ("complement-events", args.complement_events),
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
    return rc


def cmd_run(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb run",
        usage="fzb run model.pcfb --trials N [--seed S] [--backend auto|cuda|hip|omp|metal] [--json]\n"
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
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for row in rows:
            print(
                f"{row.name}  {row.estimate:.8e}  [{row.ci_low:.8e}, {row.ci_high:.8e}]  "
                f"n={row.n}  hits={row.hits}"
            )
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

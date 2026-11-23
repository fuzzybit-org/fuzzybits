"""A small JSON Schema (draft 2020-12 subset) checker for the dataset manifests.

The harness and the Python test environment carry no ``jsonschema``
dependency (``requirements-tests.txt`` is pytest, numpy, pyte and wcwidth), so
this module implements exactly the keywords the manifest schemas use:
``$ref`` into ``$defs``, ``type``, ``const``, ``enum``, ``pattern``,
``minLength``, ``minItems``, ``uniqueItems``, ``minimum``, ``exclusiveMinimum``,
``exclusiveMaximum``, ``required``, ``properties``, ``additionalProperties``,
``items``, ``oneOf``, ``anyOf``, ``allOf``, ``if``/``then``/``else`` and
``propertyNames``.

Any other keyword is an error rather than silently ignored, so a schema edit
that reaches for an unimplemented keyword fails the test that validates the
manifest instead of passing vacuously.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "$comment",
        "title",
        "description",
        "default",
        "type",
        "const",
        "enum",
        "pattern",
        "minLength",
        "minItems",
        "uniqueItems",
        "minimum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "required",
        "properties",
        "additionalProperties",
        "propertyNames",
        "items",
        "oneOf",
        "anyOf",
        "allOf",
        "if",
        "then",
        "else",
    }
)

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class SchemaError(ValueError):
    """The schema uses a keyword this checker does not implement."""


def _type_ok(value: Any, name: str) -> bool:
    if name not in _TYPES:
        raise SchemaError(f"unknown type {name!r}")
    # JSON booleans are ints in Python; keep them out of integer/number.
    if name in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[name])


def _resolve(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, Mapping) or part not in node:
            raise SchemaError(f"unresolvable $ref {ref!r}")
        node = node[part]
    return node


def _check_keywords(schema: Mapping[str, Any]) -> None:
    unknown = {key for key in schema if key not in SUPPORTED}
    if unknown:
        raise SchemaError(f"unsupported schema keyword(s): {sorted(unknown)}")
    for key in ("properties", "$defs"):
        for sub in (schema.get(key) or {}).values():
            _check_keywords(sub)
    for key in ("items", "additionalProperties", "propertyNames", "if", "then", "else", "not"):
        sub = schema.get(key)
        if isinstance(sub, Mapping):
            _check_keywords(sub)
    for key in ("oneOf", "anyOf", "allOf"):
        for sub in schema.get(key) or ():
            _check_keywords(sub)


def _walk(
    root: Mapping[str, Any],
    schema: Mapping[str, Any],
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    if "$ref" in schema:
        _walk(root, _resolve(root, str(schema["$ref"])), value, path, errors)
        return
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _walk(root, sub, value, path, errors)
    if "anyOf" in schema:
        if not any(_valid(root, sub, value) for sub in schema["anyOf"]):
            errors.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        branches = list(schema["oneOf"])
        failures = [validate_branch(root, sub, value) for sub in branches]
        matched = sum(1 for branch in failures if not branch)
        if matched == 0:
            # Report the closest branch: fewest violations, ties to the first.
            best = min(failures, key=len)
            errors.append(f"{path}: matches none of oneOf; closest: " + "; ".join(best[:6]))
        elif matched > 1:
            errors.append(f"{path}: matches {matched} of oneOf (want exactly 1)")
    if "if" in schema:
        branch = schema.get("then") if _valid(root, schema["if"], value) else schema.get("else")
        if isinstance(branch, Mapping):
            _walk(root, branch, value, path, errors)
    if "type" in schema:
        names = schema["type"]
        names = [names] if isinstance(names, str) else list(names)
        if not any(_type_ok(value, name) for name in names):
            errors.append(f"{path}: {value!r} is not {names}")
            return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: {value!r} != const {schema['const']!r}")
    if "enum" in schema and value not in list(schema["enum"]):
        errors.append(f"{path}: {value!r} not in {list(schema['enum'])}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(str(schema["pattern"]), value):
            errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {value} >= exclusiveMaximum {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems") and len(value) != len({json.dumps(v, sort_keys=True) for v in value}):
            errors.append(f"{path}: items are not unique")
        if isinstance(schema.get("items"), Mapping):
            for index, item in enumerate(value):
                _walk(root, schema["items"], item, f"{path}[{index}]", errors)
    if isinstance(value, Mapping):
        for key in schema.get("required") or ():
            if key not in value:
                errors.append(f"{path}: missing required {key!r}")
        properties = schema.get("properties") or {}
        extra_ok = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _walk(root, properties[key], item, f"{path}.{key}", errors)
                continue
            if extra_ok is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra_ok, Mapping):
                if isinstance(schema.get("propertyNames"), Mapping) and not _valid(
                    root, schema["propertyNames"], key
                ):
                    errors.append(f"{path}: property name {key!r} is not allowed")
                _walk(root, extra_ok, item, f"{path}.{key}", errors)
            elif isinstance(schema.get("propertyNames"), Mapping) and not _valid(
                root, schema["propertyNames"], key
            ):
                errors.append(f"{path}: property name {key!r} is not allowed")


def _valid(root: Mapping[str, Any], schema: Mapping[str, Any], value: Any) -> bool:
    return not validate_branch(root, schema, value)


def validate_branch(root: Mapping[str, Any], schema: Mapping[str, Any], value: Any) -> list[str]:
    """Violations of one subschema, for diagnostics and branch selection."""
    errors: list[str] = []
    _walk(root, schema, value, "$", errors)
    return errors


def validate(schema: Mapping[str, Any], document: Any) -> list[str]:
    """Return a list of ``path: message`` violations (empty when valid)."""
    _check_keywords(schema)
    errors: list[str] = []
    _walk(schema, schema, document, "$", errors)
    return errors


def validate_file(schema_path: str | Path, document_path: str | Path) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    document = json.loads(Path(document_path).read_text(encoding="utf-8"))
    return validate(schema, document)


def require_valid(schema: Mapping[str, Any], document: Any, *, what: str) -> None:
    """Raise ``ValueError`` naming every violation."""
    errors = validate(schema, document)
    if errors:
        raise ValueError(f"{what} does not match the schema:\n  " + "\n  ".join(errors))


def schema_paths(root: Path) -> Sequence[Path]:
    return sorted((root / "benchmarks" / "schemas").glob("dataset-manifest.v*.json"))

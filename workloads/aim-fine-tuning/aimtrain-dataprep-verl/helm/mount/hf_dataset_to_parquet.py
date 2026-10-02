#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Hugging Face dataset to VeRL-ready Parquet converter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, cast

from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF datasets into VeRL SFT parquet format")
    parser.add_argument("--dataset", required=True, help="Dataset identifier, e.g. openai/gsm8k")
    parser.add_argument("--config-name", default=None, help="Optional dataset config/subset name")
    parser.add_argument("--splits", default="train", help="Comma or space separated list of splits to export")
    parser.add_argument(
        "--split-percentages",
        default="",
        help="Comma or space separated percentages for named outputs (e.g., '80,10,10' or '0.8,0.1,0.1')",
    )
    parser.add_argument(
        "--split-names",
        default="",
        help="Comma or space separated output names for percentage splits (e.g., 'train,validation,test')",
    )
    parser.add_argument(
        "--split-ratios",
        default="",
        help="Mapping of split names to ratios (JSON or 'train:0.9,val:0.1')",
    )
    parser.add_argument(
        "--split-source",
        default="train",
        help="Source split to load when using percentage splits",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used to shuffle before percentage splits",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files will be written")
    parser.add_argument("--prompt-field", default=None, help="Column containing the user prompt")
    parser.add_argument("--response-field", default=None, help="Column containing the assistant response")
    parser.add_argument(
        "--prompt-alias",
        default="",
        help="Comma or space separated list of alternate prompt field names",
    )
    parser.add_argument(
        "--response-alias",
        default="",
        help="Comma or space separated list of alternate response field names",
    )
    parser.add_argument(
        "--conversation-field", default=None, help="Column containing a list of {role, content} messages"
    )
    parser.add_argument("--system-prompt", default="", help="Optional system prompt prepended to every conversation")
    parser.add_argument(
        "--system-prompt-field", default=None, help="Column containing the system prompt for each example"
    )
    parser.add_argument(
        "--system-prompt-alias",
        default="",
        help="Comma or space separated list of alternate system prompt field names",
    )
    parser.add_argument(
        "--system-prompt-template",
        default="",
        help="Template for system prompt. Use {field_name} to reference dataset fields",
    )
    parser.add_argument(
        "--prompt-template",
        default="",
        help="Template for user prompt; overrides --prompt-field when provided (use {field_name} placeholders)",
    )
    parser.add_argument("--data-source", default=None, help="Value stored in data_source; defaults to dataset name")
    parser.add_argument("--user-role", default="user", help="Role label for user turns in Q&A mode")
    parser.add_argument("--assistant-role", default="assistant", help="Role label for assistant turns in Q&A mode")
    parser.add_argument("--system-role", default="system", help="Role label for system turns")
    parser.add_argument(
        "--user-role-alias",
        default="",
        help="Comma or space separated list of alternate role labels that should map to --user-role",
    )
    parser.add_argument(
        "--assistant-role-alias",
        default="",
        help="Comma or space separated list of alternate role labels that should map to --assistant-role",
    )
    parser.add_argument(
        "--system-role-alias",
        default="",
        help="Comma or space separated list of alternate role labels that should map to --system-role",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional row cap. In percentage mode, caps the source split "
            "before splitting; in explicit-split mode, caps rows per split."
        ),
    )
    parser.add_argument(
        "--num-proc", type=int, default=1, help="Parallel workers passed to datasets.map for faster preprocessing"
    )
    parser.add_argument(
        "--preserve-fields",
        nargs="*",
        default=[],
        help="Optional list of dataset columns to copy into extra_info",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=0,
        help="When > 0, log a JSON preview head for each split",
    )
    return parser.parse_args()


def normalize_preserve_fields(raw_fields: List[str]) -> List[str]:
    fields: List[str] = []
    for item in raw_fields:
        for chunk in item.split(","):
            chunk = chunk.strip()
            if chunk:
                fields.append(chunk)
    return fields


def normalize_aliases(raw_aliases: str) -> List[str]:
    aliases: List[str] = []
    for chunk in raw_aliases.replace(",", " ").split():
        chunk = chunk.strip()
        if chunk:
            aliases.append(chunk)
    return aliases


def normalize_tokens(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    tokens: List[str] = []
    for chunk in raw_value.replace(",", " ").split():
        chunk = chunk.strip()
        if chunk:
            tokens.append(chunk)
    return tokens


def parse_percentages(raw_value: str) -> List[float]:
    tokens = normalize_tokens(raw_value)
    ensure_condition(bool(tokens), "split-percentages must include at least one value")
    values: List[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid percentage '{token}'") from exc
        ensure_condition(value > 0, f"split-percentages must be positive; got {token}")
        values.append(value)

    total = sum(values)
    needs_scale = any(value > 1.0 for value in values) or total > 1.0 + 1e-6
    if needs_scale:
        values = [value / 100.0 for value in values]
        total = sum(values)

    ensure_condition(
        abs(total - 1.0) <= 1e-6,
        "split-percentages must sum to 1.0 (or 100 when using whole percentages)",
    )
    return values


def parse_split_ratios(raw_value: str) -> Dict[str, float]:
    raw_value = raw_value.strip()
    if not raw_value:
        return {}
    if raw_value.startswith("{"):
        data = json.loads(raw_value)
        ensure_condition(isinstance(data, dict), "split-ratios must be a JSON object")
        return {str(k): float(v) for k, v in data.items()}

    ratios: Dict[str, float] = {}
    for chunk in normalize_tokens(raw_value):
        if ":" not in chunk:
            ensure_condition(False, "split-ratios entries must be name:ratio")
        name, value = chunk.split(":", 1)
        name = name.strip()
        ensure_condition(bool(name), "split-ratios entries must include a name")
        ratios[name] = float(value)
    return ratios


def ensure_condition(condition: bool, message: str) -> None:
    if not condition:
        print(message, file=sys.stderr)
        sys.exit(1)


def ensure_row_condition(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def emit_preview(split: str, dataset: Dataset, limit: int) -> None:
    if limit <= 0:
        return
    preview_count = min(limit, len(dataset))
    if preview_count <= 0:
        return

    print(f"Preview for split '{split}' (showing {preview_count} rows):")
    for idx in range(preview_count):
        row = dataset[idx]
        serialized = json.dumps(row, ensure_ascii=False)
        print(f"  [{split}:{idx}] {serialized}")


def conversation_from_fields(
    example: Dict,
    conversation_field: str | None,
    prompt_field: str | None,
    response_field: str | None,
    prompt_aliases: List[str],
    response_aliases: List[str],
    system_prompt: str,
    system_prompt_field: str | None,
    system_prompt_aliases: List[str],
    system_prompt_template: str,
    prompt_template: str,
    user_role: str,
    assistant_role: str,
    system_role: str,
    user_role_aliases: List[str],
    assistant_role_aliases: List[str],
    system_role_aliases: List[str],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []

    # Build system prompt from field, template, or static value
    system_prompt_text = str(system_prompt) if system_prompt else ""

    # If system_prompt_field is specified, try to get it from the example
    if system_prompt_field or system_prompt_aliases:
        system_key = system_prompt_field
        if system_key and system_key not in example:
            system_key = next((alias for alias in system_prompt_aliases if alias in example), system_key)
        if system_key and system_key in example:
            field_value = example.get(system_key)
            if field_value:
                system_prompt_text = str(field_value)

    # Apply template if specified
    if system_prompt_template:
        # Replace {field_name} placeholders with actual field values
        template_text = system_prompt_template
        for key, value in example.items():
            placeholder = f"{{{key}}}"
            if placeholder in template_text:
                template_text = template_text.replace(placeholder, str(value) if value is not None else "")
        system_prompt_text = template_text

    if system_prompt_text:
        messages.append({"role": str(system_role), "content": system_prompt_text})

    if conversation_field:
        raw_conversation = example.get(conversation_field)
        ensure_row_condition(
            isinstance(raw_conversation, list), f"Conversation field '{conversation_field}' must be a list"
        )
        typed_conversation = cast(List[Any], raw_conversation)
        for message in typed_conversation:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            elif isinstance(message, (list, tuple)) and len(message) == 2:
                role, content = message
            else:
                ensure_row_condition(False, "Conversation entries must be dicts or [role, content] pairs")

            ensure_row_condition(
                role is not None, f"Conversation message {message} need both role and content {typed_conversation}"
            )
            if content is None:
                content = ""

            role_str = str(role)
            if role_str in user_role_aliases:
                role_str = user_role
            elif role_str in assistant_role_aliases:
                role_str = assistant_role
            elif role_str in system_role_aliases:
                role_str = system_role

            if role_str not in (assistant_role, user_role, system_role):
                ensure_row_condition(
                    False,
                    f"Unrecognized role '{role_str}' in conversation; expected one of "
                    f"'{user_role}' (aliases: {user_role_aliases}), "
                    f"'{assistant_role}' (aliases: {assistant_role_aliases}), "
                    f"or '{system_role}' (aliases: {system_role_aliases})",
                )

            messages.append({"role": role_str, "content": str(content)})
        return messages

    prompt_key = prompt_field
    response_key = response_field
    if prompt_key not in example:
        prompt_key = next((alias for alias in prompt_aliases if alias in example), prompt_key)
    if response_key not in example:
        response_key = next((alias for alias in response_aliases if alias in example), response_key)
    # Build user prompt from field or template
    prompt = str(example.get(prompt_key, ""))
    if prompt_template:
        # Replace {field_name} placeholders with actual field values
        template_text = prompt_template
        for key, value in example.items():
            placeholder = f"{{{key}}}"
            if placeholder in template_text:
                template_text = template_text.replace(placeholder, str(value) if value is not None else "")
        prompt = template_text
    response = str(example.get(response_key, ""))
    messages.append({"role": user_role, "content": prompt})
    messages.append({"role": assistant_role, "content": response})
    return messages


def validate_required_fields(dataset: Dataset, args: argparse.Namespace, split: str) -> None:
    columns = set(dataset.column_names)
    if args.conversation_field:
        ensure_condition(
            args.conversation_field in columns,
            f"Conversation field '{args.conversation_field}' not found in split '{split}'. Available columns: {sorted(columns)}",
        )
        return

    ensure_condition(
        (args.prompt_field is not None or bool(getattr(args, "prompt_template", "")))
        and args.response_field is not None,
        "response-field must be set, and either prompt-field or --prompt-template must be provided",
    )
    # If using prompt template, do not require a prompt column.
    missing: List[str] = []
    if not getattr(args, "prompt_template", ""):
        prompt_candidates: List[str] = [args.prompt_field] + normalize_aliases(args.prompt_alias)
        if not any(candidate in columns for candidate in prompt_candidates):
            missing.append("prompt-field")
    response_candidates: List[str] = [args.response_field] + normalize_aliases(args.response_alias)
    if not any(candidate in columns for candidate in response_candidates):
        missing.append("response-field")
    ensure_condition(
        not missing,
        f"Missing required columns {missing} in split '{split}'. Available columns: {sorted(columns)}",
    )


def serialize_extra_info(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def build_extra_info(example: Dict, preserve_fields: List[str], split: str, idx: int) -> Dict[str, Any]:
    extra_info: Dict[str, Any] = {"split": split, "index": idx, "skip_reason": ""}
    for field in preserve_fields:
        if field in example:
            extra_info[field] = serialize_extra_info(example[field])
    return extra_info


def convert_example(
    example: Dict,
    idx: int,
    *,
    split_name: str,
    preserve_fields: List[str],
    conversation_kwargs: Dict[str, Any],
    data_source_value: str,
) -> Dict[str, Any]:
    try:
        conversation = conversation_from_fields(example, **conversation_kwargs)
    except ValueError as exc:
        extra_info = build_extra_info(example, preserve_fields, split_name, idx)
        extra_info["skip_reason"] = str(exc)
        return {
            "__skip__": True,
            "data_source": data_source_value,
            "messages": [],
            "extra_info": extra_info,
        }

    return {
        "__skip__": False,
        "data_source": data_source_value,
        "messages": conversation,
        "extra_info": build_extra_info(example, preserve_fields, split_name, idx),
    }


def convert_and_write_split(
    dataset: Dataset,
    split_name: str,
    args: argparse.Namespace,
    preserve_fields: List[str],
    user_role_aliases: List[str],
    assistant_role_aliases: List[str],
    system_role_aliases: List[str],
    prompt_aliases: List[str],
    response_aliases: List[str],
    system_prompt_aliases: List[str],
    data_source_value: str,
    output_dir: str,
) -> None:
    validate_required_fields(dataset, args, split_name)

    if len(dataset) == 0:
        print(f"Split '{split_name}' produced no rows; skipping write", file=sys.stderr)
        return

    conversation_kwargs = {
        "conversation_field": args.conversation_field,
        "prompt_field": args.prompt_field,
        "response_field": args.response_field,
        "prompt_aliases": prompt_aliases,
        "response_aliases": response_aliases,
        "system_prompt": args.system_prompt,
        "system_prompt_field": args.system_prompt_field,
        "system_prompt_aliases": system_prompt_aliases,
        "system_prompt_template": args.system_prompt_template,
        "prompt_template": args.prompt_template,
        "user_role": args.user_role,
        "assistant_role": args.assistant_role,
        "system_role": args.system_role,
        "user_role_aliases": user_role_aliases,
        "assistant_role_aliases": assistant_role_aliases,
        "system_role_aliases": system_role_aliases,
    }
    source_columns = dataset.column_names
    num_proc = args.num_proc if args.num_proc > 1 else None
    processed = dataset.map(
        convert_example,
        with_indices=True,
        fn_kwargs={
            "split_name": split_name,
            "preserve_fields": preserve_fields,
            "conversation_kwargs": conversation_kwargs,
            "data_source_value": data_source_value,
        },
        remove_columns=source_columns,
        num_proc=num_proc,
    )

    processed = processed.filter(lambda row: not row.get("__skip__", False))
    if "__skip__" in processed.column_names:
        processed = processed.remove_columns(["__skip__"])

    if len(processed) == 0:
        print(f"Split '{split_name}' produced no rows after processing; skipping write", file=sys.stderr)
        return

    target_path = os.path.join(output_dir, f"{split_name}.parquet")
    processed.to_parquet(target_path)
    print(f"Wrote {len(processed)} rows for split '{split_name}' to {target_path}")

    emit_preview(split_name, processed, args.preview_rows)


def main() -> None:
    args = parse_args()
    if args.max_rows is not None:
        ensure_condition(args.max_rows > 0, "--max-rows must be positive")
    ensure_condition(args.num_proc >= 1, "--num-proc must be at least 1")
    ensure_condition(args.preview_rows >= 0, "--preview-rows must be non-negative")
    preserve_fields = normalize_preserve_fields(args.preserve_fields)
    user_role_aliases = normalize_aliases(args.user_role_alias)
    assistant_role_aliases = normalize_aliases(args.assistant_role_alias)
    system_role_aliases = normalize_aliases(args.system_role_alias)
    prompt_aliases = normalize_aliases(args.prompt_alias)
    response_aliases = normalize_aliases(args.response_alias)
    system_prompt_aliases = normalize_aliases(args.system_prompt_alias)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    split_ratios = parse_split_ratios(args.split_ratios)
    if split_ratios:
        split_names = list(split_ratios.keys())
        split_percentages_raw = ",".join(str(v) for v in split_ratios.values())
    else:
        split_names = normalize_tokens(args.split_names)
        split_percentages_raw = args.split_percentages
    using_percentage_splits = bool(split_names or split_percentages_raw)

    split_tokens = [token.strip() for token in args.splits.replace(",", " ").split() if token.strip()]
    if using_percentage_splits:
        ensure_condition(
            bool(split_names) and bool(split_percentages_raw),
            "split-names and split-percentages must both be provided",
        )
    else:
        ensure_condition(len(split_tokens) > 0, "At least one split has to be specified")

    dataset_name = args.dataset
    dataset_config = args.config_name
    data_source_value = args.data_source or dataset_name

    if using_percentage_splits:
        percentages = parse_percentages(split_percentages_raw)
        ensure_condition(
            len(split_names) == len(percentages),
            "split-names and split-percentages must have the same length",
        )

        load_kwargs = {"split": args.split_source}
        if dataset_config:
            load_kwargs["name"] = dataset_config
        dataset = load_dataset(dataset_name, **load_kwargs)

        # Cap the source dataset before splitting when using percentage-based splits
        if args.max_rows is not None and len(dataset) > args.max_rows:
            dataset = dataset.select(range(args.max_rows))

        if len(dataset) == 0:
            print(
                f"Source split '{args.split_source}' produced no rows; skipping write",
                file=sys.stderr,
            )
            return

        dataset = dataset.shuffle(seed=args.split_seed)
        total_rows = len(dataset)
        counts = [int(total_rows * value) for value in percentages]
        remainder = total_rows - sum(counts)
        if counts:
            counts[-1] += remainder

        offset = 0
        for split_name, count in zip(split_names, counts):
            if count <= 0:
                print(
                    f"Split '{split_name}' produced no rows from percentages; skipping write",
                    file=sys.stderr,
                )
                continue
            subset = dataset.select(range(offset, offset + count))
            offset += count
            convert_and_write_split(
                subset,
                split_name,
                args,
                preserve_fields,
                user_role_aliases,
                assistant_role_aliases,
                system_role_aliases,
                prompt_aliases,
                response_aliases,
                system_prompt_aliases,
                data_source_value,
                output_dir,
            )
    else:
        for split in split_tokens:
            load_kwargs = {"split": split}
            if dataset_config:
                load_kwargs["name"] = dataset_config
            dataset = load_dataset(dataset_name, **load_kwargs)

            if args.max_rows is not None and len(dataset) > args.max_rows:
                dataset = dataset.select(range(args.max_rows))

            convert_and_write_split(
                dataset,
                split,
                args,
                preserve_fields,
                user_role_aliases,
                assistant_role_aliases,
                system_role_aliases,
                prompt_aliases,
                response_aliases,
                system_prompt_aliases,
                data_source_value,
                output_dir,
            )


if __name__ == "__main__":
    main()

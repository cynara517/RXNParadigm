from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dft_preprocess_agent.core.engine import WorkflowEngine
from dft_preprocess_agent.core.state import WorkflowState


ENGINE = WorkflowEngine(skills_dir=ROOT / "skills", runs_dir=ROOT / "runs")
UPLOAD_DIR = ROOT / "uploads"

SKILL_NAMES = [
    "dataset_profile",
    "grouped_correlation",
    "mutual_information",
    "ridge_dependency_graph",
    "llm_chain_decomposition",
    "feature_selection_from_chains",
    "residual_collinearity_check",
    "final_feature_export",
]

TEXT = {
    "zh": {
        "app_title": "RXNSelector",
        "sidebar_title": "RXNSelector",
        "language": "语言 / Language",
        "config_template": "配置模板",
        "config_editor": "筛选配置 YAML",
        "config_help": (
            "你可以直接在这里修改筛选参数。创建任务时，系统会保存当前 YAML "
            "作为本次任务配置。请保持合法 YAML 格式。"
        ),
        "config_caption": "配置会影响 group 解析、相关性阈值、ridge 依赖图和最终合规检查。",
        "config_guide": "配置填写说明（英文）",
        "config_guide_download": "下载配置说明",
        "dataset": "数据集",
        "dataset_help": (
            "请上传规范数据集：前 N-1 列为特征列，最后一列为被预测物。"
            "首行列名会用于特征名、group 解析和大模型反应解析，请谨慎检查。"
        ),
        "dataset_caption": "数据规范：前 N-1 列为特征，最后一列为被预测物；首行列名用于特征解析。",
        "load_demo": "载入范例",
        "create_run": "创建任务",
        "dataset_required": "请选择数据集",
        "config_invalid": "配置 YAML 无法解析：",
        "current_run": "当前任务",
        "current_data": "当前数据",
        "waiting": "等待创建任务",
        "processing": "处理中",
        "workflow": "处理流程",
        "done_suffix": "已完成",
        "metric_rows": "样本数",
        "metric_features": "原始变量",
        "metric_candidates": "候选变量",
        "metric_steps": "完成步骤",
        "metric_target": "被预测物",
        "target_column": "被预测物列名",
        "feature_name_list": "原始特征名列表",
        "group_count_table": "Group 变量数量",
        "group_feature_lists": "分 group 特征列表",
        "download_grouped_json": "下载分组特征 JSON",
        "download_grouped_csv": "下载分组特征 CSV",
        "view_results": "查看结果",
        "result_panel": "步骤结果",
        "results_for": "当前查看",
        "no_step_result": "请点击已完成步骤下方的“查看结果”。",
        "download_csv": "下载 CSV",
        "download_json": "下载 JSON",
        "download_markdown": "下载 Markdown",
        "correlation_summary": "相关性 summary",
        "high_correlation_pairs": "高相关变量对",
        "correlation_figures": "相关性图",
        "download_mi_csv": "下载 MI CSV",
        "mi_by_group": "分 group MI 得分",
        "ridge_equations": "Ridge 建模方程与 R²",
        "ridge_edges": "Ridge 依赖边",
        "rendered_markdown_report": "Markdown 公式报告",
        "chain_by_group": "分 group 链条拆解",
        "chain_summary": "链式关系总结",
        "retained_features": "保留变量",
        "removed_features": "删除变量",
        "decision_table": "筛选决策表",
        "residual_iterations": "合规复查迭代",
        "final_features": "最终特征列表",
        "create_first": "创建任务后显示结果",
        "col_step": "步骤",
        "col_status": "状态",
        "status_done": "完成",
        "status_pending": "待处理",
        "group_count": "变量数",
        "chain_decisions": "链条筛选决策",
        "residual_check": "最终相关性合规复查",
        "download_selected": "下载筛选后数据集",
        "step_labels": {
            "dataset_profile": "数据识别",
            "grouped_correlation": "相关性诊断",
            "mutual_information": "MI 评分",
            "ridge_dependency_graph": "Ridge 依赖图",
            "llm_chain_decomposition": "链条拆解",
            "feature_selection_from_chains": "链条筛选",
            "residual_collinearity_check": "合规复查",
            "final_feature_export": "导出数据",
        },
    },
    "en": {
        "app_title": "RXNSelector",
        "sidebar_title": "RXNSelector",
        "language": "Language / 语言",
        "config_template": "Config template",
        "config_editor": "Screening config YAML",
        "config_help": (
            "Edit screening parameters directly here. When a run is created, "
            "the current YAML is saved as that run's config. Keep valid YAML syntax."
        ),
        "config_caption": (
            "The config controls group parsing, correlation thresholds, ridge "
            "dependency graphs, and final compliance checks."
        ),
        "config_guide": "Config guide",
        "config_guide_download": "Download config guide",
        "dataset": "Dataset",
        "dataset_help": (
            "Upload a normalized dataset: the first N-1 columns are features and "
            "the final column is the prediction target. Header names are used for "
            "feature names, group parsing, and LLM reaction reasoning."
        ),
        "dataset_caption": (
            "Data rule: first N-1 columns are features; the final column is the target; "
            "headers are used for feature parsing."
        ),
        "load_demo": "Load demo",
        "create_run": "Create run",
        "dataset_required": "Please choose a dataset",
        "config_invalid": "Config YAML could not be parsed: ",
        "current_run": "Current run",
        "current_data": "Current data",
        "waiting": "Waiting for a run",
        "processing": "Processing",
        "workflow": "Workflow",
        "done_suffix": "done",
        "metric_rows": "Rows",
        "metric_features": "Original features",
        "metric_candidates": "Selected features",
        "metric_steps": "Completed steps",
        "metric_target": "Target",
        "target_column": "Prediction target column",
        "feature_name_list": "Original feature-name list",
        "group_count_table": "Group feature counts",
        "group_feature_lists": "Grouped feature lists",
        "download_grouped_json": "Download grouped features JSON",
        "download_grouped_csv": "Download grouped features CSV",
        "view_results": "View results",
        "result_panel": "Step results",
        "results_for": "Viewing",
        "no_step_result": "Click “View results” under a completed step.",
        "download_csv": "Download CSV",
        "download_json": "Download JSON",
        "download_markdown": "Download Markdown",
        "correlation_summary": "Correlation summary",
        "high_correlation_pairs": "High-correlation pairs",
        "correlation_figures": "Correlation figures",
        "download_mi_csv": "Download MI CSV",
        "mi_by_group": "MI scores by group",
        "ridge_equations": "Ridge equations and R²",
        "ridge_edges": "Ridge dependency edges",
        "rendered_markdown_report": "Rendered Markdown formula report",
        "chain_by_group": "Chain decomposition by group",
        "chain_summary": "Chain relationship summary",
        "retained_features": "Retained features",
        "removed_features": "Removed features",
        "decision_table": "Feature decision table",
        "residual_iterations": "Compliance iterations",
        "final_features": "Final feature list",
        "create_first": "Create a run to see results",
        "col_step": "Step",
        "col_status": "Status",
        "status_done": "Done",
        "status_pending": "Pending",
        "group_count": "Feature count",
        "chain_decisions": "Chain screening decisions",
        "residual_check": "Final correlation compliance check",
        "download_selected": "Download screened dataset",
        "step_labels": {
            "dataset_profile": "Dataset profile",
            "grouped_correlation": "Correlation diagnosis",
            "mutual_information": "MI scoring",
            "ridge_dependency_graph": "Ridge dependency graph",
            "llm_chain_decomposition": "Chain decomposition",
            "feature_selection_from_chains": "Chain selection",
            "residual_collinearity_check": "Compliance check",
            "final_feature_export": "Export dataset",
        },
    },
}


def t(lang: str, key: str):
    return TEXT.get(lang, TEXT["zh"])[key]


def skill_steps(lang: str) -> list[tuple[str, str]]:
    labels = t(lang, "step_labels")
    return [(skill_name, labels[skill_name]) for skill_name in SKILL_NAMES]


def configure_page() -> None:
    st.set_page_config(
        page_title="RXNSelector",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; }
        .stButton > button { width: 100%; border-radius: 6px; }
        .stDownloadButton > button { width: 100%; border-radius: 6px; }
        .status-chip {
            display: inline-block;
            padding: 0.16rem 0.5rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.78rem;
            font-weight: 600;
            margin-left: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def list_configs() -> list[Path]:
    return sorted((ROOT / "configs").glob("*.yaml"))


def config_guide_text() -> str:
    path = ROOT / "docs" / "config_example_annotated.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Config guide is missing."


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(uploaded_file.name).name
    destination = UPLOAD_DIR / f"{stamp}_{safe_name}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def save_client_config(config_text: str, run_id: str) -> Path:
    parsed = yaml.safe_load(config_text)
    if not isinstance(parsed, dict):
        raise ValueError("Config must be a YAML mapping/object.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{run_id}_config.yaml"
    destination.write_text(config_text, encoding="utf-8")
    return destination


def state_path() -> Path | None:
    value = st.session_state.get("state_path")
    return Path(value) if value else None


def load_state() -> WorkflowState | None:
    path = state_path()
    if path and path.exists():
        return WorkflowState.load(path)
    return None


def set_state(state: WorkflowState) -> None:
    st.session_state["state_path"] = str(state.path)


def completed_skill_names(state: WorkflowState | None) -> set[str]:
    if not state:
        return set()
    return {step["skill"] for step in state.steps if step.get("status") == "success"}


def next_step_index(state: WorkflowState | None) -> int:
    completed = completed_skill_names(state)
    for index, skill_name in enumerate(SKILL_NAMES):
        if skill_name not in completed:
            return index
    return len(SKILL_NAMES)


def load_artifact_df(state: WorkflowState | None, key: str) -> pd.DataFrame | None:
    if not state:
        return None
    path = state.artifacts.get(key)
    if not path or not Path(path).exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_artifact_json(state: WorkflowState | None, key: str) -> dict | list | None:
    if not state:
        return None
    path = state.artifacts.get(key)
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def artifact_path(state: WorkflowState | None, key: str) -> Path | None:
    if not state:
        return None
    value = state.artifacts.get(key)
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def download_file_button(label: str, path: Path, mime: str) -> None:
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=f"download_{path}",
    )


def short_feature_name(feature: str, group_name: str) -> str:
    prefix = f"{group_name}_"
    return feature[len(prefix) :] if feature.startswith(prefix) else feature


def rendered_relation(
    features: list[str],
    edges: list[dict],
    group_name: str,
    include_values: bool = True,
) -> str:
    if not features:
        return ""
    if len(features) == 1:
        return short_feature_name(features[0], group_name)
    feature_set = set(features)
    component_edges = [
        edge
        for edge in edges
        if edge.get("feature_1") in feature_set and edge.get("feature_2") in feature_set
    ]
    if not component_edges:
        return " ; ".join(short_feature_name(feature, group_name) for feature in features)
    relation_parts = []
    for edge in sorted(component_edges, key=lambda item: -float(item.get("abs_ridge_value", 0.0))):
        left = short_feature_name(edge.get("feature_1", ""), group_name)
        right = short_feature_name(edge.get("feature_2", ""), group_name)
        if include_values:
            value = float(edge.get("ridge_value", 0.0))
            arrow = f"↔<sup>{value:.3f}</sup>"
        else:
            arrow = "↔"
        relation_parts.append(f"{left} {arrow} {right}")
    return " ; ".join(relation_parts)


def normalize_chain_markdown(text: str) -> str:
    text = re.sub(
        r"\$\s*\\+leftrightarrow\^\{([^}]+)\}\s*\$",
        r"↔<sup>\1</sup>",
        text,
    )
    text = re.sub(r"\$\s*\\+leftrightarrow\s*\$", "↔", text)
    return text


def relation_has_unvalued_arrow(relation: str) -> bool:
    return "↔ " in relation or relation.endswith("↔")


def render_run_setup() -> tuple[WorkflowState | None, str]:
    lang_label = st.selectbox(
        "语言 / Language",
        ["中文", "English"],
        index=0 if st.session_state.get("lang", "zh") == "zh" else 1,
        key="language_select",
    )
    lang = "zh" if lang_label == "中文" else "en"
    st.session_state["lang"] = lang

    st.title(t(lang, "app_title"))

    configs = list_configs()
    config_labels = [path.name for path in configs]
    with st.container():
        top_col_1, top_col_2 = st.columns([1, 2])
        with top_col_1:
            selected_config_label = st.selectbox(t(lang, "config_template"), config_labels, index=0)
            selected_config = configs[config_labels.index(selected_config_label)]
            uploaded_file = st.file_uploader(
                t(lang, "dataset"),
                type=["csv", "xlsx", "xls", "parquet"],
                help=t(lang, "dataset_help"),
            )
            st.caption(t(lang, "dataset_caption"))

            action_col_1, action_col_2 = st.columns(2)
            use_demo = action_col_1.button(t(lang, "load_demo"))
            create_run = action_col_2.button(t(lang, "create_run"))
        with top_col_2:
            config_text = st.text_area(
                t(lang, "config_editor"),
                value=selected_config.read_text(encoding="utf-8"),
                height=260,
                help=t(lang, "config_help"),
                key=f"config_text_{selected_config_label}",
            )
            st.caption(t(lang, "config_caption"))

    guide = config_guide_text()
    with st.expander(t(lang, "config_guide")):
        st.markdown(guide)
        st.download_button(
            t(lang, "config_guide_download"),
            data=guide.encode("utf-8"),
            file_name="config_example_annotated.md",
            mime="text/markdown",
        )

    if use_demo:
        dataset_path = ROOT / "examples" / "00_01_data.csv"
        run_id = f"ui_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            config_path = save_client_config(config_text, run_id)
        except (ValueError, yaml.YAMLError) as exc:
            st.error(f"{t(lang, 'config_invalid')}{exc}")
            return load_state(), lang
        state = ENGINE.create_run(dataset_path, config_path, run_id=run_id)
        set_state(state)
        st.rerun()

    if create_run:
        if uploaded_file is None:
            st.error(t(lang, "dataset_required"))
        else:
            dataset_path = save_uploaded_file(uploaded_file)
            run_id = f"ui_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                config_path = save_client_config(config_text, run_id)
            except (ValueError, yaml.YAMLError) as exc:
                st.error(f"{t(lang, 'config_invalid')}{exc}")
                return load_state(), lang
            state = ENGINE.create_run(dataset_path, config_path, run_id=run_id)
            set_state(state)
            st.rerun()

    state = load_state()
    st.divider()
    if state:
        run_col, data_col = st.columns(2)
        run_col.caption(t(lang, "current_run"))
        run_col.code(state.run_id, language=None)
        data_col.caption(t(lang, "current_data"))
        data_col.write(Path(state.current_dataset_path).name)
    else:
        st.caption(t(lang, "waiting"))
    return state, lang


def run_skill(state: WorkflowState, skill_name: str, lang: str) -> None:
    with st.spinner(t(lang, "processing")):
        ENGINE.run_skill(state, skill_name)
    st.session_state["selected_result_skill"] = skill_name
    set_state(state)
    st.rerun()


def render_step_controls(state: WorkflowState | None, lang: str) -> None:
    steps = skill_steps(lang)
    st.subheader(t(lang, "workflow"))
    completed = completed_skill_names(state)
    next_index = next_step_index(state)
    columns = st.columns(len(steps))
    for index, ((skill_name, label), column) in enumerate(zip(steps, columns)):
        with column:
            done = skill_name in completed
            is_next = index == next_index
            button_label = label if not done else f"{label} {t(lang, 'done_suffix')}"
            disabled = state is None or done or not is_next
            if st.button(button_label, disabled=disabled, key=f"run_{skill_name}"):
                run_skill(state, skill_name, lang)
            view_disabled = state is None or not done
            if st.button(
                t(lang, "view_results"),
                disabled=view_disabled,
                key=f"view_{skill_name}",
            ):
                st.session_state["selected_result_skill"] = skill_name


def render_summary(state: WorkflowState | None, lang: str) -> None:
    profile = None
    if state and state.artifacts.get("dataset_profile"):
        profile_path = Path(state.artifacts["dataset_profile"])
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))

    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    col_1.metric(t(lang, "metric_rows"), profile.get("rows", "-") if profile else "-")
    col_2.metric(t(lang, "metric_features"), profile.get("feature_count", "-") if profile else "-")
    col_3.metric(t(lang, "metric_target"), profile.get("target_column", "-") if profile else "-")
    col_4.metric(t(lang, "metric_candidates"), len(state.selected_features) if state else "-")
    col_5.metric(t(lang, "metric_steps"), len(state.steps) if state else 0)


def render_dataset_profile_result(state: WorkflowState, lang: str) -> None:
    profile = load_artifact_json(state, "dataset_profile") or {}
    rows = [
        {
            t(lang, "col_step"): label,
            t(lang, "col_status"): (
                t(lang, "status_done")
                if skill in completed_skill_names(state)
                else t(lang, "status_pending")
            ),
        }
        for skill, label in skill_steps(lang)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    group_counts = profile.get("group_counts", {})
    grouped_features = profile.get("groups", {})
    if not group_counts and grouped_features:
        group_counts = {name: len(features) for name, features in grouped_features.items()}

    st.caption(t(lang, "target_column"))
    st.code(profile.get("target_column", ""), language=None)

    st.caption(t(lang, "feature_name_list"))
    st.code(repr(profile.get("feature_names", [])), language="python")

    if group_counts:
        st.caption(t(lang, "group_count_table"))
        count_df = pd.DataFrame(
            [{"group": group, "feature_count": count} for group, count in group_counts.items()]
        )
        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            st.bar_chart(pd.Series(group_counts, name=t(lang, "group_count")))
        with table_col:
            st.dataframe(count_df, use_container_width=True, hide_index=True)

    if grouped_features:
        st.caption(t(lang, "group_feature_lists"))
        grouped_json = json.dumps(grouped_features, indent=2, ensure_ascii=False)
        grouped_rows = [
            {"group": group_name, "feature_index": index, "feature": feature}
            for group_name, features in grouped_features.items()
            for index, feature in enumerate(features, start=1)
        ]
        grouped_csv = pd.DataFrame(grouped_rows).to_csv(index=False)
        download_col_1, download_col_2 = st.columns(2)
        download_col_1.download_button(
            t(lang, "download_grouped_json"),
            data=grouped_json.encode("utf-8"),
            file_name="grouped_features.json",
            mime="application/json",
        )
        download_col_2.download_button(
            t(lang, "download_grouped_csv"),
            data=grouped_csv.encode("utf-8"),
            file_name="grouped_features.csv",
            mime="text/csv",
        )
        for group_name, features in grouped_features.items():
            with st.expander(f"{group_name} ({len(features)})"):
                st.code(repr(features), language="python")


def render_grouped_correlation_result(state: WorkflowState, lang: str) -> None:
    summary_df = load_artifact_df(state, "correlation_summary")
    pairs_df = load_artifact_df(state, "high_correlation_pairs")
    if summary_df is not None:
        st.caption(t(lang, "correlation_summary"))
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        path = artifact_path(state, "correlation_summary")
        if path:
            download_file_button(t(lang, "download_csv"), path, "text/csv")
    if pairs_df is not None:
        st.caption(t(lang, "high_correlation_pairs"))
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)
        path = artifact_path(state, "high_correlation_pairs")
        if path:
            download_file_button(t(lang, "download_csv"), path, "text/csv")

    step_dir = Path(state.run_dir) / "step_02_grouped_correlation"
    figures = sorted(step_dir.glob("corr_*.svg"))
    if figures:
        st.caption(t(lang, "correlation_figures"))
        for figure_path in figures:
            with st.expander(figure_path.stem.replace("corr_", "")):
                st.image(str(figure_path))
                download_file_button(t(lang, "download_json").replace("JSON", "SVG"), figure_path, "image/svg+xml")


def render_mutual_information_result(state: WorkflowState, lang: str) -> None:
    mi_df = load_artifact_df(state, "mi_scores_by_group")
    mi_path = artifact_path(state, "mi_scores_by_group")
    if mi_df is None:
        return
    if mi_path:
        download_file_button(t(lang, "download_mi_csv"), mi_path, "text/csv")
    st.caption(t(lang, "mi_by_group"))
    for group_name, group_df in mi_df.groupby("group", sort=False):
        with st.expander(f"{group_name} ({len(group_df)})", expanded=True):
            display_df = group_df.sort_values("mi_rank_in_group")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.download_button(
                f"{t(lang, 'download_csv')} - {group_name}",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name=f"mi_scores_{group_name}.csv",
                mime="text/csv",
                key=f"download_mi_{group_name}",
            )
            top_df = display_df.head(30)
            if not top_df.empty:
                st.bar_chart(top_df.set_index("feature")["mi_score"])


def render_ridge_dependency_result(state: WorkflowState, lang: str) -> None:
    report = load_artifact_json(state, "ridge_dependency_report")
    if not isinstance(report, dict):
        return
    json_path = artifact_path(state, "ridge_dependency_report")
    md_path = artifact_path(state, "ridge_dependency_report_md")
    download_cols = st.columns(2)
    if json_path:
        with download_cols[0]:
            download_file_button(t(lang, "download_json"), json_path, "application/json")
    if md_path:
        with download_cols[1]:
            download_file_button(t(lang, "download_markdown"), md_path, "text/markdown")

    if md_path:
        st.caption(t(lang, "rendered_markdown_report"))
        st.markdown(
            normalize_chain_markdown(md_path.read_text(encoding="utf-8")),
            unsafe_allow_html=True,
        )

    st.caption(t(lang, "ridge_equations"))
    for group_name, group_data in report.get("groups", {}).items():
        with st.expander(group_name, expanded=True):
            components = group_data.get("raw_components", [])
            edges = group_data.get("ridge_edges", [])
            if components:
                st.caption(t(lang, "chain_summary"))
                relation_lines = []
                for index, component in enumerate(components, start=1):
                    stored_relation = component.get("rendered_relation", "")
                    relation = stored_relation if stored_relation and not relation_has_unvalued_arrow(stored_relation) else rendered_relation(
                        component.get("features", []),
                        edges,
                        group_name,
                        include_values=True,
                    )
                    relation_lines.append(f"{index}. {relation}")
                st.markdown(
                    normalize_chain_markdown("\n\n".join(relation_lines)),
                    unsafe_allow_html=True,
                )
            formula_rows = []
            for feature, formula in group_data.get("formulas", {}).items():
                formula_rows.append(
                    {
                        "feature": feature,
                        "variable_type": group_data.get("feature_types", {}).get(feature, ""),
                        "ridge_r2": formula.get("ridge_r2"),
                        "is_multicollinear": formula.get("is_multicollinear"),
                        "contributor_count": len(formula.get("contributors", {})),
                        "formula": formula.get("formula", ""),
                    }
                )
            if formula_rows:
                st.dataframe(pd.DataFrame(formula_rows), use_container_width=True, hide_index=True)
            edge_rows = group_data.get("ridge_edges", [])
            if edge_rows:
                st.caption(t(lang, "ridge_edges"))
                st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)


def render_chain_decomposition_result(state: WorkflowState, lang: str) -> None:
    report = load_artifact_json(state, "llm_chain_decomposition")
    if not isinstance(report, dict):
        return
    json_path = artifact_path(state, "llm_chain_decomposition")
    md_path = artifact_path(state, "llm_chain_decomposition_md")
    download_cols = st.columns(2)
    if json_path:
        with download_cols[0]:
            download_file_button(t(lang, "download_json"), json_path, "application/json")
    if md_path:
        with download_cols[1]:
            download_file_button(t(lang, "download_markdown"), md_path, "text/markdown")

    if md_path:
        st.caption(t(lang, "rendered_markdown_report"))
        st.markdown(
            normalize_chain_markdown(md_path.read_text(encoding="utf-8")),
            unsafe_allow_html=True,
        )

    st.caption(t(lang, "chain_by_group"))
    for group_name, group_data in report.get("groups", {}).items():
        with st.expander(group_name, expanded=True):
            chains = group_data.get("chains", [])
            rendered_lines = []
            for index, chain in enumerate(chains, start=1):
                fallback_relation = rendered_relation(
                    chain.get("features", []),
                    chain.get("ridge_edges", []),
                    group_name,
                    include_values=True,
                )
                stored_relation = chain.get("rendered_relation", "")
                relation = stored_relation if stored_relation and not relation_has_unvalued_arrow(stored_relation) else fallback_relation
                rendered_lines.append(f"{index}. {relation}")
            if rendered_lines:
                st.caption(t(lang, "chain_summary"))
                st.markdown(
                    normalize_chain_markdown("\n\n".join(rendered_lines)),
                    unsafe_allow_html=True,
                )
            for chain in chains:
                st.markdown(f"**{chain.get('chain_id')}** · `{chain.get('importance')}`")
                if chain.get("ridge_edges"):
                    st.dataframe(pd.DataFrame(chain["ridge_edges"]), use_container_width=True, hide_index=True)
                retained = chain.get("retained_features", [])
                removed = chain.get("removed_features", [])
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(t(lang, "retained_features"))
                    if retained:
                        st.dataframe(pd.DataFrame(retained), use_container_width=True, hide_index=True)
                with col_b:
                    st.caption(t(lang, "removed_features"))
                    if removed:
                        st.dataframe(pd.DataFrame(removed), use_container_width=True, hide_index=True)


def render_feature_selection_result(state: WorkflowState, lang: str) -> None:
    decision_df = load_artifact_df(state, "feature_decision_table")
    if decision_df is not None:
        st.caption(t(lang, "decision_table"))
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        path = artifact_path(state, "feature_decision_table")
        if path:
            download_file_button(t(lang, "download_csv"), path, "text/csv")
    selected = load_artifact_json(state, "initial_selected_features")
    if isinstance(selected, dict):
        st.caption(t(lang, "retained_features"))
        st.code(repr(selected.get("features", [])), language="python")


def render_residual_result(state: WorkflowState, lang: str) -> None:
    iterations_df = load_artifact_df(state, "residual_collinearity_iterations")
    if iterations_df is not None:
        st.caption(t(lang, "residual_iterations"))
        st.dataframe(iterations_df, use_container_width=True, hide_index=True)
        path = artifact_path(state, "residual_collinearity_iterations")
        if path:
            download_file_button(t(lang, "download_csv"), path, "text/csv")
    final_features = load_artifact_json(state, "final_features")
    if isinstance(final_features, dict):
        st.caption(t(lang, "final_features"))
        st.code(repr(final_features.get("features", [])), language="python")
        path = artifact_path(state, "final_features")
        if path:
            download_file_button(t(lang, "download_json"), path, "application/json")
    matrix_path = artifact_path(state, "final_correlation_matrix")
    graph_path = artifact_path(state, "final_correlation_graph")
    if matrix_path:
        download_file_button(t(lang, "download_csv"), matrix_path, "text/csv")
    if graph_path:
        st.image(str(graph_path))


def render_final_export_result(state: WorkflowState, lang: str) -> None:
    selected_dataset_path = artifact_path(state, "selected_dataset")
    if selected_dataset_path:
        st.download_button(
            t(lang, "download_selected"),
            data=selected_dataset_path.read_bytes(),
            file_name=selected_dataset_path.name,
            mime="text/csv",
        )
    report_path = artifact_path(state, "final_report")
    if report_path:
        st.markdown(report_path.read_text(encoding="utf-8"))
        download_file_button(t(lang, "download_markdown"), report_path, "text/markdown")


def render_step_result_panel(state: WorkflowState | None, lang: str) -> None:
    st.subheader(t(lang, "result_panel"))
    if not state:
        st.info(t(lang, "create_first"))
        return
    selected_skill = st.session_state.get("selected_result_skill")
    completed = completed_skill_names(state)
    if selected_skill not in completed:
        completed_ordered = [skill for skill in SKILL_NAMES if skill in completed]
        selected_skill = completed_ordered[-1] if completed_ordered else None
    if not selected_skill:
        st.info(t(lang, "no_step_result"))
        return

    label = dict(skill_steps(lang)).get(selected_skill, selected_skill)
    st.caption(f"{t(lang, 'results_for')}: {label}")
    renderers = {
        "dataset_profile": render_dataset_profile_result,
        "grouped_correlation": render_grouped_correlation_result,
        "mutual_information": render_mutual_information_result,
        "ridge_dependency_graph": render_ridge_dependency_result,
        "llm_chain_decomposition": render_chain_decomposition_result,
        "feature_selection_from_chains": render_feature_selection_result,
        "residual_collinearity_check": render_residual_result,
        "final_feature_export": render_final_export_result,
    }
    renderers[selected_skill](state, lang)


def main() -> None:
    configure_page()
    state, lang = render_run_setup()

    render_summary(state, lang)
    render_step_controls(state, lang)
    render_step_result_panel(state, lang)


if __name__ == "__main__":
    main()

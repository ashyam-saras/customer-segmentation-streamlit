import json
import os
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from dbtsl import SemanticLayerClient


# Initialize client with caching
@st.cache_resource
def get_client(environment_id: int, auth_token: str, host: str):
    """Initialize and cache the Semantic Layer client."""
    return SemanticLayerClient(
        environment_id=environment_id,
        auth_token=auth_token,
        host=host,
    )


@st.cache_data
def get_all_metrics(environment_id: int, auth_token: str, host: str):
    """Load and cache all metrics - opens own session."""
    client = get_client(environment_id, auth_token, host)
    with client.session():
        metrics = client.metrics()
    return metrics


@st.cache_data(show_spinner="Loading dimension values...")
def get_dimension_values_cached(environment_id: int, auth_token: str, host: str, metrics: tuple, dimension: str):
    """Load and cache dimension values - opens own session."""
    client = get_client(environment_id, auth_token, host)
    try:
        with client.session():
            dim_values = client.dimension_values(metrics=list(metrics), group_by=dimension)
            values = list(dim_values.to_pydict().values())[0]

        # Format dates for metric_time dimensions
        if "metric_time" in dimension.lower() and values:
            # Extract just the date part (YYYY-MM-DD) from datetime strings
            values = [str(v)[:10] if len(str(v)) > 10 else str(v) for v in values]

        # Sort values for better UX
        try:
            # Try to sort naturally (works for strings, numbers, dates)
            return sorted(values, reverse=True)  # Most recent dates first
        except:
            return values
    except Exception as e:
        # Return empty list if there's an error - will fallback to text input
        st.warning(f"Could not load values for {dimension}: {str(e)}")
        return []


@st.cache_data
def compile_sql_cached(
    environment_id: int,
    auth_token: str,
    host: str,
    metrics: tuple,
    group_by: tuple,
    where_clause: list[str],
):
    """Compile SQL query without executing it - opens own session."""
    client = get_client(environment_id, auth_token, host)
    params = {"metrics": list(metrics)}

    if group_by:
        params["group_by"] = list(group_by)

    # Only add where clause if it's not empty
    if where_clause:
        params["where"] = where_clause

    with client.session():
        sql = client.compile_sql(**params)

    return sql


@st.cache_data
def execute_query_cached(
    environment_id: int,
    auth_token: str,
    host: str,
    metrics: tuple,
    group_by: tuple,
    where_clause: list[str],
):
    """Execute query and return results with compiled SQL - opens own session."""
    client = get_client(environment_id, auth_token, host)
    params = {"metrics": list(metrics)}

    if group_by:
        params["group_by"] = list(group_by)

    # Only add where clause if it's not empty
    if where_clause:
        params["where"] = where_clause

    with client.session():
        # Get results
        table = client.query(**params)
        df = table.to_pandas()

        # Get compiled SQL
        sql = client.compile_sql(**params)

    return df, sql


# Segment persistence functions
SEGMENTS_FILE = "segments.json"


def load_segments() -> List[Dict[str, Any]]:
    """Load saved segments from JSON file."""
    if os.path.exists(SEGMENTS_FILE):
        with open(SEGMENTS_FILE, "r") as f:
            data = json.load(f)
            return data.get("segments", [])
    return []


def save_segment(name: str, params: Dict[str, Any], where_conditions: List[Dict[str, Any]]):
    """Save a segment to JSON file."""
    segments = load_segments()
    segment = {
        "name": name,
        "params": params,
        "where_conditions": where_conditions,
        "created_at": datetime.now().isoformat(),
    }
    segments.append(segment)

    with open(SEGMENTS_FILE, "w") as f:
        json.dump({"segments": segments}, f, indent=2)


def delete_segment(segment_name: str):
    """Delete a segment from JSON file."""
    segments = load_segments()
    segments = [s for s in segments if s["name"] != segment_name]

    with open(SEGMENTS_FILE, "w") as f:
        json.dump({"segments": segments}, f, indent=2)


def build_where_clause(conditions: List[Dict[str, Any]]) -> str:
    """Build a where clause from conditions list."""
    if not conditions:
        return ""

    clause_parts = []
    for i, condition in enumerate(conditions):
        if not condition.get("dimension") or condition.get("value") is None:
            continue

        dimension = condition["dimension"]
        operator = condition["operator"]
        value = condition["value"]

        # Handle date formatting for metric_time dimensions
        if "metric_time" in dimension.lower() and len(str(value)) > 10:
            # Extract just the date part (YYYY-MM-DD) from datetime strings
            value = str(value)[:10]

        # Build the condition string
        condition_str = f"{{{{ Dimension('{dimension}') }}}} {operator} '{value}'"

        # Add logic operator (AND/OR) before this condition if not the first
        if i > 0 and condition.get("logic"):
            clause_parts.append(condition["logic"])

        clause_parts.append(condition_str)

    return [" ".join(clause_parts)]


def separate_metric_and_dimension_filters(conditions: List[Dict[str, Any]], metrics_list: List[str]):
    """Separate conditions into metric filters and dimension filters."""
    metric_filters = []
    dimension_filters = []

    for condition in conditions:
        field = condition.get("dimension", "").replace("📊 ", "").replace("📁 ", "")
        if field in metrics_list:
            metric_filters.append({**condition, "field": field})
        else:
            dimension_filters.append({**condition, "field": field})

    return metric_filters, dimension_filters


def build_dimension_where_clause(dimension_filters: List[Dict[str, Any]]) -> List[str]:
    """Build where clause only for dimensions (for semantic layer)."""
    if not dimension_filters:
        return []

    clause_parts = []
    for i, condition in enumerate(dimension_filters):
        if not condition.get("field") or condition.get("value") is None:
            continue

        field = condition["field"]
        operator = condition["operator"]
        value = condition["value"]

        # Handle date formatting for metric_time dimensions
        if "metric_time" in field.lower() and len(str(value)) > 10:
            value = str(value)[:10]

        # Build the condition string
        condition_str = f"{{{{ Dimension('{field}') }}}} {operator} '{value}'"

        # Add logic operator (AND/OR) before this condition if not the first
        if i > 0 and condition.get("logic"):
            clause_parts.append(condition["logic"])

        clause_parts.append(condition_str)

    return [" ".join(clause_parts)]


def build_metric_where_clause(metric_filters: List[Dict[str, Any]]) -> str:
    """Build SQL WHERE clause for metrics (for outer query)."""
    if not metric_filters:
        return ""

    clause_parts = []
    for i, condition in enumerate(metric_filters):
        if not condition.get("field") or condition.get("value") is None:
            continue

        field = condition["field"]
        operator = condition["operator"]
        value = condition["value"]

        # Build the condition string (standard SQL, not Jinja)
        # Try to detect if value is numeric
        try:
            numeric_value = float(value)
            condition_str = f"{field} {operator} {numeric_value}"
        except (ValueError, TypeError):
            # If not numeric, treat as string
            condition_str = f"{field} {operator} '{value}'"

        # Add logic operator (AND/OR) before this condition if not the first
        if i > 0 and condition.get("logic"):
            clause_parts.append(condition["logic"])

        clause_parts.append(condition_str)

    return " ".join(clause_parts)


def execute_bigquery_with_metric_filters(
    project_id: str,
    credentials_dict: Dict[str, Any],
    compiled_sql: str,
    metric_where_clause: str,
):
    """Execute wrapped SQL on BigQuery with metric filters."""
    from google.cloud import bigquery
    from google.oauth2 import service_account

    # Initialize BigQuery client
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = bigquery.Client(credentials=credentials, project=project_id)

    # Wrap compiled SQL with metric filters
    final_sql = f"""SELECT *
FROM (
{compiled_sql}
) AS semantic_layer_query
WHERE {metric_where_clause}"""

    # Execute and return DataFrame
    df = client.query(final_sql).to_dataframe()
    return df, final_sql


def initialize_session_state():
    """Initialize session state variables."""
    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = []
    if "selected_group_by" not in st.session_state:
        st.session_state.selected_group_by = []
    if "where_conditions" not in st.session_state:
        st.session_state.where_conditions = []
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "compiled_sql" not in st.session_state:
        st.session_state.compiled_sql = None
    if "bq_credentials" not in st.session_state:
        st.session_state.bq_credentials = None
    if "bq_project_id" not in st.session_state:
        st.session_state.bq_project_id = None


@st.dialog("Save Segment")
def save_segment_dialog(selected_metrics, selected_group_by, where_conditions):
    """Dialog for saving a segment."""
    segment_name = st.text_input("Segment Name", placeholder="Enter a name for this segment")

    if st.button("Save", type="primary", width="stretch"):
        if segment_name:
            params = {"metrics": selected_metrics}
            if selected_group_by:
                params["group_by"] = selected_group_by

            where_clause = build_where_clause(where_conditions)
            if where_clause:
                params["where"] = where_clause

            save_segment(segment_name, params, where_conditions)
            st.success(f"✅ Segment '{segment_name}' saved successfully!")
            st.session_state.show_save_dialog = False
            st.rerun()
        else:
            st.error("Please enter a segment name")


def main():
    st.set_page_config(page_title="dbt Semantic Layer - Customer Segmentation", layout="wide")

    initialize_session_state()

    # Initialize dialog state
    if "show_save_dialog" not in st.session_state:
        st.session_state.show_save_dialog = False

    st.title("🎯 dbt Semantic Layer - Customer Segmentation")

    # Sidebar: Configuration and Saved Segments
    with st.sidebar:
        with st.expander("⚙️ dbt Semantic Layer Configuration", expanded=True):
            # dbt Semantic Layer credentials - load from secrets or allow manual override
            default_env_id = st.secrets.get("dbt", {}).get("environment_id", 70403103993055)
            default_auth_token = st.secrets.get("dbt", {}).get("auth_token", "")
            default_host = st.secrets.get("dbt", {}).get("host", "bm843.semantic-layer.us1.dbt.com")

            environment_id = st.number_input(
                "Environment ID", value=int(default_env_id), step=1, format="%d", help="Your dbt Cloud Environment ID"
            )

            auth_token = st.text_input(
                "Auth Token",
                value=default_auth_token,
                type="password",
                help="Your dbt Cloud API token",
            )

            host = st.text_input("Host", value=default_host, help="Your dbt Semantic Layer host")

        with st.expander("🔑 BigQuery Credentials", expanded=False):
            st.caption("Required for metric-based WHERE filters")

            # Try to load credentials from secrets first, then from file
            if st.session_state.bq_credentials is None:
                # Check if credentials exist in secrets
                if "bigquery" in st.secrets:
                    try:
                        st.session_state.bq_credentials = dict(st.secrets["bigquery"])
                        st.session_state.bq_project_id = st.secrets["bigquery"].get("project_id")
                        st.success("✅ Loaded credentials from secrets")
                    except Exception as e:
                        st.warning(f"Could not load credentials from secrets: {str(e)}")
                # Fallback to local file
                else:
                    default_creds_file = "bigquery_service_account.json"
                    if os.path.exists(default_creds_file):
                        try:
                            with open(default_creds_file, "r") as f:
                                default_creds = json.load(f)
                                st.session_state.bq_credentials = default_creds
                                st.session_state.bq_project_id = default_creds.get("project_id")
                                st.success("✅ Loaded credentials from file")
                        except Exception as e:
                            st.warning(f"Could not load default credentials: {str(e)}")

            # File uploader for service account
            uploaded_file = st.file_uploader(
                "Upload Service Account JSON", type=["json"], help="Upload your BigQuery service account credentials"
            )

            if uploaded_file is not None:
                try:
                    credentials_dict = json.load(uploaded_file)
                    st.session_state.bq_credentials = credentials_dict
                    st.session_state.bq_project_id = credentials_dict.get("project_id")
                    st.success(f"✅ Credentials loaded for project: {st.session_state.bq_project_id}")
                except Exception as e:
                    st.error(f"Error loading credentials: {str(e)}")

            # Display current status
            if st.session_state.bq_credentials:
                st.info(f"📊 Project: {st.session_state.bq_project_id}")
            else:
                st.warning("⚠️ No credentials loaded")

        st.divider()
        st.header("💾 Saved Segments")

        saved_segments = load_segments()

        if not saved_segments:
            st.info("No saved segments yet.")
        else:
            for segment in saved_segments:
                with st.expander(f"📊 {segment['name']}"):
                    st.write("**Metrics:**", ", ".join(segment["params"].get("metrics", [])))
                    if "group_by" in segment["params"]:
                        st.write("**Group By:**", ", ".join(segment["params"]["group_by"]))
                    if "where" in segment["params"]:
                        where_display = segment["params"]["where"]
                        if isinstance(where_display, list):
                            where_display = where_display[0] if where_display else ""
                        st.write("**Where:**", where_display)
                    st.caption(f"Created: {segment['created_at'][:10]}")

                    col_load, col_delete = st.columns(2)
                    with col_load:
                        if st.button("📥 Load", key=f"load_{segment['name']}", width="stretch"):
                            st.session_state.selected_metrics = segment["params"].get("metrics", [])
                            st.session_state.selected_group_by = segment["params"].get("group_by", [])
                            st.session_state.where_conditions = segment.get("where_conditions", [])
                            st.success(f"Loaded: {segment['name']}")
                            st.rerun()

                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_{segment['name']}", width="stretch"):
                            delete_segment(segment["name"])
                            st.success(f"Deleted: {segment['name']}")
                            st.rerun()

    # Main layout: Query Builder on Left, Results on Right
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("Query Builder")

        # Metrics and Group By in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Select Metrics")
            # Load all metrics (cached)
            all_metrics = get_all_metrics(environment_id, auth_token, host)
            metric_names = sorted([m.name for m in all_metrics])

            # Ensure selected_metrics are still valid options
            valid_selected_metrics = [m for m in st.session_state.selected_metrics if m in metric_names]

            selected_metrics = st.multiselect(
                "Metrics",
                options=metric_names,
                default=valid_selected_metrics,
                help="Select one or more metrics to query",
            )
            # Update session state after widget interaction
            st.session_state.selected_metrics = selected_metrics

        with col2:
            st.subheader("Select Group By")
            # Get available dimensions based on selected metrics
            available_dimensions = []

            if selected_metrics:
                # Get common dimensions across selected metrics
                metric_objects = [m for m in all_metrics if m.name in selected_metrics]
                if metric_objects:
                    dimension_sets = [set(d.name for d in m.dimensions) for m in metric_objects]
                    common_dimensions = set.intersection(*dimension_sets) if dimension_sets else set()
                    available_dimensions = sorted(list(common_dimensions))  # Already sorted

            # Ensure selected_group_by are still valid options
            valid_selected_group_by = [gb for gb in st.session_state.selected_group_by if gb in available_dimensions]

            selected_group_by = st.multiselect(
                "Group By (Dimensions)",
                options=available_dimensions,
                default=valid_selected_group_by,
                help="Select dimensions to group by",
            )
            # Update session state after widget interaction
            st.session_state.selected_group_by = selected_group_by

        # Where Conditions Builder
        st.subheader("Where Conditions")

        # Build combined list of dimensions and metrics for WHERE conditions
        available_fields = []
        if available_dimensions:
            available_fields.extend([f"📁 {dim}" for dim in available_dimensions])
        if selected_metrics:
            available_fields.extend([f"📊 {metric}" for metric in selected_metrics])

        if available_fields:
            # Add condition button
            col_add, col_clear = st.columns([1, 1])
            with col_add:
                if st.button("➕ Add Condition", width="stretch"):
                    st.session_state.where_conditions.append(
                        {
                            "dimension": available_fields[0] if available_fields else "",
                            "operator": "=",
                            "value": None,
                            "logic": "AND",
                        }
                    )
            with col_clear:
                if st.button("🗑️ Clear All", width="stretch"):
                    st.session_state.where_conditions = []

            # Display existing conditions
            operators = ["=", "!=", ">", "<", ">=", "<="]
            logic_options = ["AND", "OR"]

            conditions_to_remove = []

            for idx, condition in enumerate(st.session_state.where_conditions):
                # All rows use same column structure for alignment
                cols = st.columns([0.6, 2.5, 0.5, 2.5, 0.5])

                # Logic operator (AND/OR) - shown at START for all except first condition
                with cols[0]:
                    if idx > 0:
                        logic = st.selectbox(
                            "Logic",
                            options=logic_options,
                            index=(
                                logic_options.index(condition["logic"]) if condition["logic"] in logic_options else 0
                            ),
                            key=f"logic_{idx}",
                            label_visibility="collapsed",
                        )
                        condition["logic"] = logic
                    else:
                        st.write("")  # Empty space for first row to maintain alignment

                with cols[1]:
                    # Field selector (dimensions + metrics)
                    # Ensure current condition value is in available_fields
                    current_field = condition["dimension"]
                    if current_field not in available_fields:
                        # Try to add prefix if missing
                        clean_field = current_field.replace("📊 ", "").replace("📁 ", "")
                        if clean_field in selected_metrics:
                            current_field = f"📊 {clean_field}"
                        elif clean_field in available_dimensions:
                            current_field = f"📁 {clean_field}"

                    field = st.selectbox(
                        "Field",
                        options=available_fields,
                        index=(available_fields.index(current_field) if current_field in available_fields else 0),
                        key=f"dim_{idx}",
                        label_visibility="collapsed",
                    )
                    condition["dimension"] = field

                with cols[2]:
                    # Operator selector
                    operator = st.selectbox(
                        "Operator",
                        options=operators,
                        index=operators.index(condition["operator"]) if condition["operator"] in operators else 0,
                        key=f"op_{idx}",
                        label_visibility="collapsed",
                    )
                    condition["operator"] = operator

                with cols[3]:
                    # Value selector - different logic for metrics vs dimensions
                    clean_field = field.replace("📊 ", "").replace("📁 ", "")
                    is_metric = field.startswith("📊 ")

                    if is_metric:
                        # For metrics, use text input (typically numeric values)
                        value = st.text_input(
                            "Value",
                            value=condition["value"] or "",
                            key=f"val_{idx}",
                            label_visibility="collapsed",
                            placeholder="Enter numeric value",
                        )
                        condition["value"] = value
                    else:
                        # For dimensions, fetch dimension values
                        if clean_field and selected_metrics:
                            # Convert metrics list to tuple for caching
                            dim_values = get_dimension_values_cached(
                                environment_id, auth_token, host, tuple(selected_metrics), clean_field
                            )
                            # Handle different data types
                            if dim_values and len(dim_values) > 0:
                                # Convert to strings for display
                                dim_values_str = [str(v) for v in dim_values]
                                value = st.selectbox(
                                    "Value",
                                    options=dim_values_str,
                                    index=(
                                        dim_values_str.index(str(condition["value"]))
                                        if condition["value"] and str(condition["value"]) in dim_values_str
                                        else 0
                                    ),
                                    key=f"val_{idx}",
                                    label_visibility="collapsed",
                                )
                                condition["value"] = value
                            else:
                                # Fallback to text input if no values available
                                value = st.text_input(
                                    "Value",
                                    value=condition["value"] or "",
                                    key=f"val_{idx}",
                                    label_visibility="collapsed",
                                    placeholder="Enter value manually",
                                )
                                condition["value"] = value

                with cols[4]:
                    # Remove button
                    if st.button("❌", key=f"remove_{idx}"):
                        conditions_to_remove.append(idx)

            # Remove marked conditions
            for idx in sorted(conditions_to_remove, reverse=True):
                st.session_state.where_conditions.pop(idx)
                st.rerun()

        else:
            st.info("Select metrics to enable where conditions")

        # Action Buttons
        st.divider()
        col_compile, col_query, col_save = st.columns([1, 1, 1])

        with col_compile:
            compile_button = st.button("📝 Compile SQL", width="stretch")

        with col_query:
            query_button = st.button("🔍 Query", type="primary", width="stretch")

        with col_save:
            save_button = st.button("💾 Save Segment", width="stretch")

    # Right Column: Results with Tabs
    with right_col:
        st.header("Results")

        # Tabs for Query Results and Compiled SQL
        tab1, tab2 = st.tabs(["📊 Query Results", "📝 Compiled SQL"])

        with tab1:
            if st.session_state.query_results is not None:
                st.metric("Total Rows", len(st.session_state.query_results))
                st.dataframe(st.session_state.query_results, width="stretch", height=700, hide_index=True)
            else:
                st.info("Click 'Query' to execute and view results")

        with tab2:
            if st.session_state.compiled_sql:
                st.code(st.session_state.compiled_sql, language="sql", height=700)
            else:
                st.info("Click 'Compile SQL' or 'Query' to view compiled SQL")

    # Compile SQL Only (outside of column contexts to avoid session conflicts)
    if compile_button:
        if not selected_metrics:
            st.error("Please select at least one metric")
        else:
            # Separate metric and dimension filters
            metric_filters, dimension_filters = separate_metric_and_dimension_filters(
                st.session_state.where_conditions, selected_metrics
            )

            # Build where clause for dimensions only
            dimension_where_clause = build_dimension_where_clause(dimension_filters)

            # Compile SQL using cached function
            with st.spinner("Compiling SQL..."):
                try:
                    sql = compile_sql_cached(
                        environment_id,
                        auth_token,
                        host,
                        tuple(selected_metrics),
                        tuple(selected_group_by) if selected_group_by else (),
                        dimension_where_clause,
                    )

                    # If there are metric filters, wrap the SQL
                    if metric_filters:
                        metric_where_clause = build_metric_where_clause(metric_filters)
                        final_sql = f"""SELECT *
FROM (
{sql}
) AS semantic_layer_query
WHERE {metric_where_clause}"""
                        st.session_state.compiled_sql = final_sql
                        st.success("SQL compiled successfully with metric filters!")
                    else:
                        st.session_state.compiled_sql = sql
                        st.success("SQL compiled successfully!")

                    st.rerun()
                except Exception as e:
                    st.error(f"Error compiling SQL: {str(e)}")

    # Execute Query (outside of column contexts to avoid session conflicts)
    if query_button:
        if not selected_metrics:
            st.error("Please select at least one metric")
        else:
            # Separate metric and dimension filters
            metric_filters, dimension_filters = separate_metric_and_dimension_filters(
                st.session_state.where_conditions, selected_metrics
            )

            # Build where clause for dimensions only
            dimension_where_clause = build_dimension_where_clause(dimension_filters)

            # Execute query
            with st.spinner("Executing query..."):
                try:
                    if metric_filters:
                        # Path 1: Metric filters exist - use BigQuery execution
                        if not st.session_state.bq_credentials:
                            st.error(
                                "❌ BigQuery credentials required for metric filters. Please upload credentials in the sidebar."
                            )
                        else:
                            # First, compile SQL with only dimension filters
                            compiled_sql = compile_sql_cached(
                                environment_id,
                                auth_token,
                                host,
                                tuple(selected_metrics),
                                tuple(selected_group_by) if selected_group_by else (),
                                dimension_where_clause,
                            )

                            # Build metric WHERE clause
                            metric_where_clause = build_metric_where_clause(metric_filters)

                            # Execute on BigQuery with wrapped SQL
                            df, final_sql = execute_bigquery_with_metric_filters(
                                st.session_state.bq_project_id,
                                st.session_state.bq_credentials,
                                compiled_sql,
                                metric_where_clause,
                            )

                            st.session_state.query_results = df
                            st.session_state.compiled_sql = final_sql

                            st.success(
                                f"✅ Query executed via BigQuery with metric filters! Retrieved {len(df)} rows."
                            )
                            st.rerun()
                    else:
                        # Path 2: No metric filters - use standard semantic layer execution
                        df, sql = execute_query_cached(
                            environment_id,
                            auth_token,
                            host,
                            tuple(selected_metrics),
                            tuple(selected_group_by) if selected_group_by else (),
                            dimension_where_clause,
                        )

                        st.session_state.query_results = df
                        st.session_state.compiled_sql = sql

                        st.success(f"✅ Query executed via Semantic Layer! Retrieved {len(df)} rows.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")

    # Save Segment - Open dialog
    if save_button:
        if not selected_metrics:
            st.error("Please configure a query before saving")
        else:
            save_segment_dialog(selected_metrics, selected_group_by, st.session_state.where_conditions)


if __name__ == "__main__":
    main()

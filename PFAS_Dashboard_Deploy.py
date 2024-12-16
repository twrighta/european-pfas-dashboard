import pandas as pd
import polars as pl
import numpy as np
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly_express as px
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.figure_factory as ff
import warnings
from flask_caching import Cache
from functools import lru_cache

warnings.simplefilter("ignore")  # Ignore warnings

# Define Styles of various dashboard components
plot_style = {"margins": {"t": 30,
                          "b": 20,
                          "l": 10,
                          "r": 10},
              "height": "90vh",
              "width": "90vh",
              "opacity": 0.75}

radar_plot_style = {"margins": {"t": 50,
                                "b": 50,
                                "l": 100,
                                "r": 50},
                    "height": "90vh"}  # Large left margin to make space for longer labels

heading_style = {"font-weight": "bold",
                 "padding": "5px"}

subheading_style = {"font-weight": 500,
                    "padding": "2.5px",
                    "font-size": "20px"}

sidecol_value_style = {"font-weight": 600,
                       "padding": "2.5px",
                       "font-size": "24px",
                       "color": "#1f77b4"}

button_style = {"border-radius": "10px",
                "padding": "5px"}

# Instantiate Dashapp
dash._dash_renderer._set_react_version('18.2.0')  # This stops dash-mantine components breaking

app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.FLATLY])
server = app.server

cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'})  # simple in-memory cache

# Loaded into Github
github_path = 'https://raw.githubusercontent.com/twrighta/european-pfas-dashboard/main/processed_PFAs_2019_2024.parquet'

# Function to load data
@cache.memoize(timeout=120)
def load_data(path):
    df_pandas = pl.read_parquet(path).to_pandas()
    return df_pandas


# Load dataframe
df = load_data(github_path)

# Create a dictionary of all unique values in most columns. Used for filtering chart
ALL_LABELS = {
    f"ALL_{col}": df[col].dropna().unique().tolist() + ["All"]
    for col in df.columns if col not in ["lat", "lon", "value"]
}

# Create location types list
MEASUREMENT_LOCATION_TYPES = [x for x in ALL_LABELS["ALL_measurement location type"] if x != "All"]

# Consistent plot colouring:
# Hex colour discrete maps (CDMs) where required (PFAS type):
pfa_type_cdm = {"Perfluoroalkyl PFAs": "#1f77b4",
                "Polyfluoroalkyl PFAs": "#ff7f0e",
                "Unclassified": "#2ca02c"}

location_type_cdm = {"Oceanic": "#1f77b4",
                     "Terrestrial": "#2ca02c",
                     "Unknown": "#ff7f0e"}

colour_discrete_sequence = ["#1f77b4", "#ff7f0e", "#2ca02c", "#DFFC03", "#FC2003"]

# Create Sidebar Html Div Element
sidebar = html.Div([
    # Heading
    dbc.Row([
        html.H1("Filters",
                style=heading_style)
    ],
        style={"height": "10vh"}),  # total vh: 10
    # Filtering Dropdowns 2 x 2 grid - Year; Country; Substance; Measurement Location Type
    dbc.Row([
        dbc.Col([
            html.P("Year",
                   style=subheading_style),
            dcc.Dropdown(options=ALL_LABELS["ALL_year"],
                         value="2023",
                         id="sidebar-year-dropdown",
                         placeholder="Select a year",
                         style=button_style),

            html.P("Country",
                   style=subheading_style),
            dcc.Dropdown(options=ALL_LABELS["ALL_country"],
                         value="United Kingdom",
                         id="sidebar-country-dropdown",
                         placeholder="Select a country",
                         style=button_style)
        ]),
        dbc.Col([
            html.P("PFA type",
                   style=subheading_style),
            dcc.Loading(overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                        type="circle",
                        children=[
                            dcc.Dropdown(options=ALL_LABELS["ALL_PFA type"],
                                         value="All",
                                         id="sidebar-pfa-type-dropdown",
                                         placeholder="Select a PFA type",
                                         style=button_style)]),

            html.P("Location Type",
                   style=subheading_style),
            dcc.Dropdown(options=ALL_LABELS["ALL_measurement location type"],
                         value="All",
                         id="sidebar-location-type-dropdown",
                         placeholder="Select a location type",
                         style=button_style)
        ])

    ],
        style={"height": "25vh"}),

    # Count or Sum of PFAs switch
    dbc.Row([
        html.P("Select measurement method:",
               style={"display": "flex",
                      "alignItems": "left"}),
        dcc.RadioItems(options=["Sum", "Count"],
                       value="Count",
                       inline=True,
                       labelStyle={"padding": "5x 10px"},
                       id="sum-count-radio",
                       style={"display": "flex",
                              "alignItems": "left",
                              "gap": "15px"})
    ],
        style={"height": "5vh",
               "display": "flex",
               "alignItems": "left"}),
    dbc.Row([
        html.Hr()
    ],
        style={"height": "5vh",
               "padding": "10px"}),

    # Headline Figures
    dbc.Row([
        dbc.Col([
            html.H3("No. Unique PFAS",
                    style=subheading_style),
            html.H4(id="sidebar-unique-pfas",
                    style=sidecol_value_style)
        ],
            style={"backgroundColor": "#c7ddec",
                   "borderRadius": "35px",
                   "margin": "10px"}),
        dbc.Col([
            html.H3("Average Sample Mass (ng)",
                    style=subheading_style),
            html.H4(id="sidebar-avg-pfa-value",
                    style=sidecol_value_style)
        ],
            style={"backgroundColor": "#ffdfc3",
                   "borderRadius": "35px",
                   "margin": "10px"}
        )
    ],
        style={"height": "10vh"}),
    dbc.Row([
        dbc.Col([
            html.H3("No. Samples (No. Studies)",
                    style=subheading_style),
            html.H4(id="sidebar-samples-studies",
                    style=sidecol_value_style)
        ],
            style={"backgroundColor": "#ffdfc3",
                   "borderRadius": "35px",
                   "margin": "10px"}),
        dbc.Col([
            html.H3("Percentage Terrestrial Samples",
                    style=subheading_style),
            html.H4(id="sidebar-pct-terrestrial",
                    style=sidecol_value_style)
        ],
            style={"backgroundColor": "#c7ddec",
                   "borderRadius": "35px",
                   "margin": "10px"})
    ],
        style={"height": "10vh"}),  # Total vh 65

    dbc.Row([
        html.Hr(style={"borderTop": "1px solid dee2e6"})
    ],
        style={"height": "3vh"}),  # Total vh 70

    dbc.Row([
        dcc.Graph("sidebar-pfa-type-pie-fig")
    ],
        style={"height": "30vh"})  # Total vh 100

],
    style={"height": "100vh"})

# Create Content #
content = html.Div([
    # Top row, 50% vh, Split into 2 columns
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.RadioItems(id="map-choice",
                               options=["Heatmap", "Points"],
                               value="Heatmap",
                               inline=True,
                               className="ml-3",
                               labelClassName="ml-3",
                               style={"display": "flex",
                                      "justifyContent": "left",
                                      "gap": "15px"},
                               labelStyle={"paddingLeft": "10px"}),

                dcc.Graph(id="map-fig",
                          style={"height": "47.5vh"})
            ],
                style={"height": "50vh",
                       "padding": "5px",
                       "display": "flex",
                       "flexDirection": "column",
                       "justifyContent": "center"})

        ],
            width=6, className="px-2"),
        dbc.Col([
            html.Div([
                dcc.Graph(id="content-count-sum-radar-fig",
                          style={"height": "45vh",
                                 "justifyContent": "center"})
            ],
                style={"height": "50vh",
                       "padding": "2.5px",
                       "display": "flex",
                       "justifyContent": "center"})
        ],
            width=6, className="px-2")
    ],
        style={"height": "50vh"}),

    # Second row - 30% vh, split into 3 columns
    dbc.Row([
        dbc.Col([
            html.Div([
                # Scatter plot
                dcc.Graph(id='content-count-sum-scatter-fig',
                          style={"height": "30vh"})
            ],
                style={"height": "30vh",
                       "padding": "5px",
                       "display": "flex",
                       "justifyContent": "center"})
        ],
            width=6, className="px-2"),
        dbc.Col([
            html.Div([

                dcc.RadioItems(options=["Pie", "Table"],
                               value="Pie",
                               inline=True,
                               id="pie-table-choice",
                               className="ml-3",
                               labelClassName="ml-3",
                               style={"display": "flex",
                                      "justifyContent": "left",
                                      "gap": "15px"},
                               labelStyle={"paddingLeft": "10px"}),
                dcc.Graph(id="pie-table-fig",
                          style={"height": "30vh"})
            ],
                style={"height": "30vh",
                       "padding": "5px",
                       "display": "flex",
                       "flexDirection": "column",
                       "justifyContent": "center"})  # Add padding
        ],
            width=6, className="px-2")
    ],
        style={"height": "30vh"}),

    # Bottom (3rd) row - 20vh, Split into 2 columns
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.RadioItems(id="year-month-radio",
                               options=["year", "month"],
                               value="year",
                               inline=True,
                               className="ml-3",
                               labelClassName="ml-3",
                               style={
                                   "display": "flex",
                                   "justifyContent": "left",
                                   "gap": "15px"},
                               labelStyle={"paddingLeft": "10px"}),
                # Line chart
                dcc.Graph(id="line-fig",
                          style={"height": "20vh"})
            ],
                style={"height": "20vh",
                       "padding": "5px",
                       "display": "flex",
                       "flexDirection": "column",
                       "justifyContent": "center"})
        ],
            width=6, className="px-2"),
        dbc.Col([
            html.Div([
                # Bar - Top cities
                dcc.Graph(id="cities-bar-fig",
                          style={"height": "20vh"})
            ],
                style={"height": "20vh",
                       "padding": "5px",
                       "justifyContent": "center",
                       "display": "flex"})
        ],
            width=6, className="px-2")
    ],
        style={"height": "20vh"})
],
    style={"height": "100vh"})

# Create app layout within a dbc.Container. Wrapped in MantineProvider to use Mantine components
app.layout = dmc.MantineProvider(
    dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3, className="bg-light"),
            dbc.Col(content, width=9, className='bg-dark bg-gradient')
        ],
            style={"height": "100vh"})
    ],
        fluid=True,
        style={"height": "100vh"}),
    forceColorScheme="light",
    id="mantine-provider"
)


# Helper function: Filter dataframe by sidebar filters
@cache.memoize(timeout=300)
def filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd):
    # Dictionary of dropdown filters
    filters = {
        "year": year_dd,
        "country": country_dd,
        "PFA type": pfa_type_dd,
        "measurement location type": location_type_dd
    }

    # Construct  mask using only active filters
    conditions = [
        (df[col] == value) for col, value in filters.items() if value != "All" and value is not None
    ]

    # Combine conditions using bitwise AND
    if conditions:
        combined_mask = conditions[0]
        for condition in conditions[1:]:
            combined_mask &= condition
        # Apply mask and return filtered dataframe
        return df[combined_mask]
    else:
        # No filtering needed; return original dataframe
        return df


# Only for filtering a dataframe to a location, not the PFA type info
@cache.memoize(timeout=300)
def filter_df_helper_location(df, year_dd, country_dd):
    # Dictionary of default dropdown filters
    filters = {"year": year_dd,
               "country": country_dd}

    # Construct the mask using only active filters - using list comprehension for speed.
    conditions = [
        (df[col] == value) for col, value in filters.items() if value != "All" and value is not None
    ]

    # Combine conditions using bitwise AND
    if conditions:
        combined_mask = conditions[0]
        for condition in conditions[1:]:
            combined_mask &= condition
        # Apply mask and return filtered dataframe
        return df[combined_mask]
    else:
        # No filtering needed; return original dataframe
        return df


# Sidebar headline figures
@lru_cache(maxsize=128)
@app.callback([Output("sidebar-unique-pfas", "children"),
               Output("sidebar-avg-pfa-value", "children"),
               Output("sidebar-samples-studies", "children"),
               Output("sidebar-pct-terrestrial", "children")],
              Input("sidebar-year-dropdown", "value"),
              Input("sidebar-country-dropdown", "value"),
              Input("sidebar-pfa-type-dropdown", "value"),
              Input("sidebar-location-type-dropdown", "value"))
def update_sidebar_headlines(year_dd, country_dd, pfa_type_dd, location_type_dd):
    filtered_df = filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd)[["substance", "value",
                                                                                            "study_id",
                                                                                            "Oceanic Terrestrial Flag"]]
    len_filtered_df = len(filtered_df)

    if len_filtered_df == 0:
        return 0, 0, "0 (0)", 0

    # Unique PFAs
    unique_pfas = len(filtered_df["substance"].unique())

    # Average value of PFAs
    avg_pfa_value = np.nanmedian(filtered_df["value"]) if len(filtered_df["value"].dropna()) > 0 else 0

    # Total samples taken (studies)
    total_studies = str(len(filtered_df["study_id"].unique()))
    samples_studies = str(len_filtered_df) + ' (' + total_studies + ')'

    # Location type ratio - Percentage on land
    try:
        pct_on_land = round(
            len(filtered_df[filtered_df["Oceanic Terrestrial Flag"] == "Terrestrial"]) / (len_filtered_df) * 100, 3)
    except ZeroDivisionError:
        pct_on_land = 0
    return unique_pfas, avg_pfa_value, samples_studies, pct_on_land


# Sidebar pie figure
@app.callback(Output("sidebar-pfa-type-pie-fig", "figure"),
              Input("sum-count-radio", "value"),
              Input("sidebar-year-dropdown", "value"),
              Input("sidebar-country-dropdown", "value"),
              Input("sidebar-pfa-type-dropdown", "value"),
              Input("sidebar-location-type-dropdown", "value"))
def update_sidebar_pie(count_sum, year_dd, country_dd, pfa_type_dd, location_type_dd):
    # Filter dataframe using helper function
    filtered_df = filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd)[["PFA type", "value"]]

    # Check if filtered_df is empty - return errored pie.
    if filtered_df.empty:
        empty_pie = px.pie(names=["No data"],
                           values=[1],
                           title="<b>No data for these selected filters<b>",
                           template="seaborn",
                           opacity=plot_style["opacity"])
        empty_pie.update_layout(margin=plot_style["margins"],
                                plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='rgba(0, 0, 0, 0)')

        return empty_pie

    aggregation_method = "sum" if count_sum == "Sum" else "count"

    grouped = filtered_df.groupby(by="PFA type", as_index=False).agg({"value": aggregation_method})

    pie = px.pie(grouped,
                 names="PFA type",
                 values="value",
                 color="PFA type",
                 title=f"<b>PFAS by {aggregation_method}<b>",
                 template="seaborn",
                 hover_name="PFA type",
                 color_discrete_map=pfa_type_cdm)
    pie.update_layout(margin=plot_style["margins"],
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)'
                      )
    return pie


# Content - Scatterplot - x-axis is number of unique PFAs; y-axis is average values; size is total summed value;
@app.callback(Output("content-count-sum-scatter-fig", "figure"),
              Input("sidebar-year-dropdown", "value"),
              Input("sidebar-country-dropdown", "value"),
              Input("sidebar-pfa-type-dropdown", "value"),
              Input("sidebar-location-type-dropdown", "value"))
def update_content_scatter(year_dd, country_dd, pfa_type_dd, location_type_dd):
    filtered_df = filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd)[["study_id",
                                                                                            "Oceanic Terrestrial Flag",
                                                                                            "substance", "value"]]

    if filtered_df.empty:
        scatter = px.pie(names=["No data"],
                         values=[1],
                         title="<b>No data for these selected filters<b>",
                         template="seaborn",
                         opacity=plot_style["opacity"])
        scatter.update_layout(margin=plot_style["margins"],
                              plot_bgcolor='rgba(0, 0, 0, 0)',
                              paper_bgcolor='rgba(0, 0, 0, 0)')
        scatter.update_yaxes(showgrid=False, zeroline=False)
        scatter.update_xaxes(showgrid=False, zeroline=False)
        return scatter

    # Group by study_id, then oceanic/terrestrial etc. Aggregate substance by number of unique substances;
    filtered_grouped = filtered_df.groupby(by=["study_id", "Oceanic Terrestrial Flag"], as_index=False).agg({
        "substance": lambda x: x.nunique(),
        "value": "median",
        "study_id": "size"}).reset_index()

    # Reset Column names
    filtered_grouped.columns = ["study_id", "Oceanic Terrestrial Flag", "num_substances", "avg_value", "num_samples"]

    # Create Scatter otherwise
    scatter = px.scatter(filtered_grouped,
                         x="num_substances",
                         y="avg_value",
                         color="Oceanic Terrestrial Flag",
                         size="num_samples",  # This hasnt worked correctly - theyre all same size.
                         title=f"<b>PFAS Count and Sample Mass per Study<b>",
                         log_y=True,
                         size_max=20,
                         opacity=plot_style["opacity"],
                         template="seaborn",
                         color_discrete_map=location_type_cdm)

    scatter.update_layout(margin=plot_style["margins"],
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          xaxis_title="Number of Unique PFAS",
                          yaxis_title="Average Sample Mass (ng)",
                          legend_title_text="Sample Location Type"
                          )
    scatter.update_xaxes(showgrid=False, zeroline=False, showspikes=True)
    scatter.update_yaxes(showgrid=False, zeroline=False, showspikes=True)  # Remove grid lines
    return scatter


# Radar plot - Location type points; coloured by PFA type
@app.callback(Output("content-count-sum-radar-fig", "figure"),
              Input("sidebar-year-dropdown", "value"),
              Input("sidebar-country-dropdown", "value"),
              Input("sum-count-radio", "value"))
def update_content_radar(year_dd, country_dd, count_sum):
    filtered_df = filter_df_helper_location(df, year_dd, country_dd)[["PFA type", "measurement location type", "value"]]

    if filtered_df.empty:
        fig = px.pie(names=["No data"],
                     values=[1],
                     title="<b>No data for these selected filters<b>",
                     template="seaborn")
        fig.update_layout(margin=plot_style["margins"],
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)')
        return fig

    # Arguments required for radar chart - needs to be grouped df
    aggregation_method = "sum" if count_sum == "Sum" else "count"

    grouped_df = filtered_df.groupby(by=["PFA type", "measurement location type"],
                                     as_index=False).agg({"value": aggregation_method})

    fig = px.bar_polar(grouped_df,
                       r="value",
                       theta="measurement location type",
                       color="PFA type",
                       log_r=True,
                       template="seaborn",
                       barnorm="percent",  # Make it percentage
                       title="<b>Occurrence per Location Type and PFAS Type<b>",
                       hover_name="PFA type",
                       hover_data={"PFA type": False,
                                   "value": True,
                                   "measurement location type": False},
                       color_discrete_map=pfa_type_cdm)
    fig.update_layout(margin=radar_plot_style["margins"],
                      plot_bgcolor='rgba(0, 0, 0, 0.2)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      template="seaborn",
                      legend_title_text="PFAS Type",
                      polar=dict(bgcolor="rgba(255, 255, 255, 0.2)",
                                 angularaxis=dict(gridcolor="rgba(0,0,0,0.3)",
                                                  tickfont=dict(size=15)),
                                 # Make the axes of the polar chart more transparent
                                 radialaxis=dict(gridcolor="rgba(0,0,0,0.1)",
                                                 tickfont=dict(size=11.5),
                                                 tickvals=[1, 10, 100, 1000, 10000, 100000, 1000000],
                                                 ticktext=["1, 10, 100, 1000, 10k, 100k, 1M"])))
    return fig


# Map - show all points, or show as heatmap
@app.callback(
    Output("map-fig", "figure"),
    [Input("map-choice", "value"),
     Input("sidebar-year-dropdown", "value"),
     Input("sidebar-country-dropdown", "value"),
     Input("sidebar-pfa-type-dropdown", "value"),
     Input("sidebar-location-type-dropdown", "value")]
)
def content_map(map_choice, year_dd, country_dd, pfa_type_dd, location_type_dd):
    proc_df = filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd)[["PFA type",
                                                                                        "measurement location type",
                                                                                        "value", "lat", "lon"]].dropna(
        subset=["value", "lat", "lon"])
    try:
        # Get coordinates for maps:
        if country_dd != "All":
            center_lat = np.median(proc_df["lat"])
            center_lon = np.median(proc_df["lon"])
        else:
            center_lat = 46
            center_lon = 1

        if map_choice == "Heatmap":
            scatter_hex = ff.create_hexbin_mapbox(data_frame=proc_df,
                                                  lat="lat",
                                                  lon="lon",
                                                  nx_hexagon=100,
                                                  zoom=5,
                                                  labels={"color": "Sample Count"},
                                                  title="<b>Locations of PFAS Samples (>0.1% of Country)<b>",
                                                  mapbox_style="carto-positron",
                                                  opacity=0.3,
                                                  min_count=len(proc_df) / 1000,
                                                  center=dict(lat=center_lat, lon=center_lon))
            scatter_hex.update_layout(margin=plot_style["margins"],
                                      plot_bgcolor='rgba(0, 0, 0, 0)',
                                      paper_bgcolor='rgba(0, 0, 0, 0)')
            scatter_hex.update_geos(showlakes=True, lakecolor="Blue",
                                    showrivers=True, rivercolor="Blue",
                                    showcountries=True)

            return scatter_hex
        elif map_choice == "Points":
            scatter_mapbox = px.scatter_mapbox(
                proc_df,
                lat="lat",
                lon="lon",
                color="PFA type",
                mapbox_style="carto-positron",
                center={"lat": center_lat,
                        "lon": center_lon},
                zoom=5,
                title="<b>Locations of PFAS Samples<b>",
                opacity=plot_style["opacity"],
                color_discrete_map=pfa_type_cdm
            )

            scatter_mapbox.update_layout(margin=plot_style["margins"],
                                         plot_bgcolor='rgba(0, 0, 0, 0)',
                                         paper_bgcolor='rgba(0, 0, 0, 0)')
            scatter_mapbox.update_geos(showlakes=True, lakecolor="Blue",
                                       showrivers=True, rivercolor="Blue",
                                       showcountries=True)

            return scatter_mapbox
    # KeyError occurs when there is no value associated with the given filters. Just show a blank map
    except KeyError:
        return px.scatter_mapbox().update_layout(
            title="<b>No Data Available<b>",
            margin=dict(l=0, r=0, t=0, b=0),
            mapbox_style="carto-positron",
            center={"lat": 0,
                    "lon": 0},
            zoom=8,
            opacity=plot_style["opacity"])


# Table / Pie - top PFA chemicals. With a 'view as pie' button
@app.callback(Output("pie-table-fig", "figure"),
              [Input("pie-table-choice", "value"),
               Input("sum-count-radio", "value"),
               Input("sidebar-year-dropdown", "value"),
               Input("sidebar-country-dropdown", "value"),
               Input("sidebar-pfa-type-dropdown", "value"),
               Input("sidebar-location-type-dropdown", "value")])
def table_pie_top_pfas(chart_type, count_sum, year_dd, country_dd, pfa_type_dd, location_type_dd):
    proc_df = filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd)[
        ["substance", "PFA type", "value"]]

    # Convert substance to category column for more efficient grouping
    proc_df["substance"] = proc_df["substance"].astype("category")

    # Length of df to show (n) - show 5 if not table, otherwise show top 10 ordered
    if chart_type != "Table":
        length_shown = 5 if len(df["substance"].unique()) > 5 else len(df["substance"].unique())
    else:
        length_shown = 10 if len(df["substance"].unique()) > 10 else len(df["substance"].unique())

    if count_sum == "Count":
        grouped = proc_df.groupby(by=["substance", "PFA type"], as_index=False).count().nlargest(columns="value",
                                                                                                 keep="all",
                                                                                                 n=length_shown)
    else:
        grouped = proc_df.groupby(by=["substance", "PFA type"], as_index=False).sum().nlargest(columns="value",
                                                                                               keep="all",
                                                                                               n=length_shown)
    # If empty:
    if grouped.empty:
        fig = px.pie(names=["No data"],
                     values=[1],
                     title="<b>No data for these selected filters<b>",
                     template="seaborn")

        fig.update_layout(margin=plot_style["margins"],
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          legend_title_text="Name")
        return fig

    if chart_type == "Pie":
        fig = px.pie(data_frame=grouped,
                     names="substance",
                     values="value",
                     color="substance",
                     title=f"<b>Top {str(len(grouped))} PFAS by {count_sum}<b>",
                     template="seaborn",
                     color_discrete_sequence=colour_discrete_sequence,
                     hover_name="substance")
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          margin=plot_style["margins"],
                          legend_title_text="PFAS Name")
        return fig

    # View as table:
    elif chart_type == "Table":
        fig = ff.create_table(grouped[["substance", "PFA type", "value"]],
                              height_constant=20,
                              colorscale=[[0, "#1f77b4"], [0.5, "#8fbad9"], [1, "#ffffff"]])
        return fig


# line - vh20, w30 - Value/count each year / month
@app.callback(Output("line-fig", "figure"),
              [Input("year-month-radio", "value"),
               Input("sum-count-radio", "value"),
               Input("sidebar-year-dropdown", "value"),
               Input("sidebar-country-dropdown", "value"),
               Input("sidebar-pfa-type-dropdown", "value"),
               Input("sidebar-location-type-dropdown", "value")])
def content_line(year_month, count_sum, year_dd, country_dd, pfa_type_dd, location_type_dd):
    proc_df = filter_df_helper(df, year_dd, country_dd, pfa_type_dd, location_type_dd)[["year", "month", "value",
                                                                                        "PFA type"]]
    str_year_month = str(year_month)

    month_reindex = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                     "November", "December", "Unknown"]  # Use to manually reorder when using months to logical order..

    if proc_df.empty or proc_df is None:
        fig = px.pie(names=["No data"],
                     values=[1],
                     title="<b>No data for these selected filters<b>",
                     template="seaborn",
                     opacity=plot_style["opacity"])
        fig.update_layout(margin=plot_style["margins"],
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)')
        return fig

    aggregation_method = "sum" if count_sum == "Sum" else "count"

    grouped_df = proc_df[[year_month, "PFA type", "value"]].groupby(by=[str_year_month, "PFA type"],
                                                                    as_index=False).agg({"value": aggregation_method})

    if year_month == "month":
        # Ensure the 'month' column is categorical with the proper month order
        grouped_df[year_month] = pd.Categorical(
            grouped_df[year_month],
            categories=month_reindex,
            ordered=True
        )
        # Sort by corrected month order
        grouped_df = grouped_df.sort_values(by=year_month)

    # Create figure
    fig = px.line(data_frame=grouped_df,
                  x=year_month,
                  y="value",
                  color="PFA type",
                  title=f"<b>PFAS by {year_month}<b>",
                  template="seaborn",
                  log_y=True,
                  color_discrete_map=pfa_type_cdm)
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      margin=plot_style["margins"],
                      yaxis_title=f"PFAS {count_sum}",
                      xaxis_title="Year",
                      legend_title_text="PFAS Type")
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor="rgba(0,0,0,0.1)")  # Remove grid lines
    return fig


# bar - top cities in country filter. vh20, w30
@app.callback(Output("cities-bar-fig", "figure"),
              [Input("sum-count-radio", "value"),
               Input("sidebar-year-dropdown", "value"),
               Input("sidebar-country-dropdown", "value")])
def top_cities_country(count_sum, year_dd, country_dd):
    filtered_df = filter_df_helper_location(df, year_dd, country_dd)[["city", "value"]]
    # Convert city to categorical column = faster grouping for high cardinality columns
    filtered_df["city"] = filtered_df["city"].astype("category")

    if not filtered_df.empty:
        if count_sum == "Count":
            grouped = filtered_df.groupby(by="city", as_index=False).count().nlargest(n=5, columns="value", keep="all")
        elif count_sum == "Sum":
            grouped = filtered_df.groupby(by="city", as_index=False).sum().nlargest(n=5, columns="value", keep="all")

        fig = px.histogram(data_frame=grouped,
                           x="city",
                           y="value",
                           barmode="group",
                           title=f"<b>Worst Locations by PFAS {count_sum}<b>",
                           template="seaborn",
                           hover_name="city",
                           color_discrete_sequence=colour_discrete_sequence)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          margin=plot_style["margins"],
                          yaxis_title=f"PFAS {count_sum}",
                          xaxis_title="Location")
        fig.update_xaxes(showgrid=False, zeroline=False)  # Remove zero lines
        fig.update_yaxes(showgrid=False, zeroline=False)  # Remove grid lines

        return fig
    else:
        fig = px.pie(names=["No data"],
                     values=[1],
                     title="<b>No data for these selected filters<b>",
                     template="seaborn")

        fig.update_layout(margin=plot_style["margins"],
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        return fig


# Run app
if __name__ == "__main__":
    app.run(debug=True)

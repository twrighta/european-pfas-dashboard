# Imports
import numpy as np
import pandas as pd
import json
from global_land_mask import globe
import warnings

warnings.simplefilter("ignore")

df_pd = pd.read_parquet('C:/Users/tomwr/Datascience/Datasets/Tabular/PFAs/pdh_data.parquet')

# Tidy up the below code into a function with explanations of each step.
def preprocess_pfas(df):
    expanded_rows = []  # Instantiate empty list to put JSON lists into

    filtered_df = df[df["pfas_values"] != "[]"]  # Filter to where pfas_values is not empty.

    # Iterate through each row in the dataframe
    for index, row in filtered_df.iterrows():
        parsed_json = json.loads(row["pfas_values"])  # Load each a rows "pfas_values" column value as a json.
        df_expanded = pd.json_normalize(parsed_json)  # Explode list of jsons into separate dictionaries then
        # each value pair put into separate rows with a col for each unique key in all jsons

        df_expanded["original_index"] = index  # Set original_idnex value to the index in df
        expanded_rows.append(df_expanded)  # Append each row sequentially below each other to create df

    # Concatenate all expanded rows into one - This is just the json columns
    df_expanded_all = pd.concat(expanded_rows, ignore_index=True)

    # Ensure filtered_df has an index column to join on
    filtered_df = filtered_df.reset_index().rename(columns={"index": "original_index"})

    # Left join the json columns back onto the original dataset
    rejoined = pd.merge(df_expanded_all, filtered_df, how="left", left_on="original_index", right_on="original_index")

    # Remove unnecessary columns
    rejoined = rejoined.drop(columns=["cas_id", "unit_x", "original_index", "source_text", "source_url",
                                      "dataset_name", "pfas_values", "details"])

    # Rename columns
    rejoined.rename(columns={"unit_y": "measurement units",
                             "matrix": "measurement location type",
                             "dataset_id": "study_id"},
                    inplace=True)

    # Reorder columns
    rejoined = rejoined[
        ["study_id", "year", "date", "name", "category", "lat", "lon", "city", "country", "type", "sector",
         "measurement location type", "substance", "value", "measurement units"]]

    # Fix dtypes
    rejoined["year"] = rejoined["year"].astype("int64")

    # Replace years set to 0 or 1900 with 2024
    rejoined.replace({"year": {0: 2024,
                               1900: 2024}},
                     inplace=True)

    # Create month column
    rejoined["month"] = rejoined["date"].astype(str).str.split("-").str[1]
    rejoined.replace({"month": {"01": "January",
                                "02": "February",
                                "03": "March",
                                "04": "April",
                                "05": "May",
                                "06": "June",
                                "07": "July",
                                "08": "August",
                                "09": "September",
                                "10": "October",
                                "11": "November",
                                "12": "December",
                                np.nan: "Unknown"}},
                     inplace=True)

    # Return
    return rejoined

# Create stage 1 of processed dataframe
preprocessed = preprocess_pfas(df_pd)

# Apply conversion
def convert_to_ng_per_l(row):
    if row['measurement location type'] == 'Terrestrial' and row['measurement units'] == 'ng/kg':
        # Convert ng/kg to ng/L
        return row['value'] * 1.3  # Using 1.3kg/L as a generic bulk density for 'soil'
    return row['value']


def update_unit(row):
    if row['measurement location type'] == 'Terrestrial' and row['measurement units'] == 'ng/kg':
        return 'ng/L'  # Update the unit
    return row['measurement units']  # Leave unchanged


# Identify if point is in land or sea
def ocean_sea_flag(df, lat_col, lon_col):
    for index, row in df.iterrows():
        try:
            df.at[index, "Oceanic Terrestrial Flag"] = globe.is_ocean((df.at[index, lat_col]), (df.at[index, lon_col]))
        except:
            df.at[index, "Oceanic Terrestrial Flag"] = "Unknown"

    # Replace np.False_ and np.True_ with meaningful
    df["Oceanic Terrestrial Flag"].replace(np.False_, "Terrestrial", inplace=True)
    df["Oceanic Terrestrial Flag"].replace(np.True_, "Oceanic", inplace=True)

    # Apply conversions to 'value' and 'unit' columns - get a consistent ng/l units
    df['value'] = df.apply(convert_to_ng_per_l, axis=1)
    df['measurement units'] = df.apply(update_unit, axis=1)
    df["measurement units"] = "ng/L"

    return df

# Create stage 2 of processed df.
preprocessed_and_flagged = ocean_sea_flag(preprocessed, "lat", "lon")


#Function to add a PFA group column - using PFas.com and PubMed
def add_pfa_group(df):

    # These are Perfluoroalykl substances
    non_polymers_PERFs = ["PFAA", "PFOA", "PFSA", "PFOS", "PFBS", "PASF", "POSF", "PBSF", "FASA", "PFHxDA", "PFHpS",
                          "PFHxl", "PFHxS", "POF", "PFAL", "PAF", "PFNAL", "PFNS", "PFPeA", "PFNA", "PFDA", "PFDS",
                          "PFDoS", "PFBA", "PFDoA", "PFDoTS", "PFCH", "PFOSF", "PFCP", "PFCPMSF", "PFOSAE", "PFMSA",
                          "PFUnDA", "PFU(n)DA", "PFDoDA", "PFTrDA", "PFTeDA", "PFPeDA", "PFHxDA",
                          "Pentadecafluorooctanoic", "heptadecafluorooctane-1-sulfonic", "PASF", "PFAL",
                          "PFdiCA", "PFdiSA", "PFECA", "FASA", "PFPiA", "PFHx", "PFHxDA", "PFHxA",
                          "γ-ω-perfluoro C10-20-thiols", "GenX", "N-ME-FOSA", "Perfluorododecyltrifluorosilane",
                          "PFHpA", "PFHxPA", "PFECH", "PFD", "PFTrS", "PFESA", "3-(Perfluorohexyl)propyl acrylate",
                          "PFDPA", "Hexafluoropropylene oxide dimer acid", "H,1H,2H,2H-perfluorododecanol",
                          "Propionic acid, tetrafluoro-3-(trifluoromethoxy)- (8CI)", "PFECHS",
                          "2,2,3,3,4,4-Hexafluoro-4-(trifluoromethoxy)butanoic acid",
                          "4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-Pentadecafluorodecanoic acid",
                          "Acetic acid, 2,2-difluoro-2-[[2,2,4,5-tetrafluoro-5-(trifluoromethoxy)-1,3-dioxolan-4-yl]oxy]-, ammonium salt (1:1)",
                          "C604", "PFMOBA", "PFECA", "PfHxDA",
                          "1-Octanesulfonic acid, 1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8-heptadecafluoro-, potassium salt (8CI,9CI)",
                          "3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12-Heneicosafluorododecanal",
                          "Tetradecanoic acid, heptacosafluoro- (9CI)",
                          "Octanoic acid, pentadecafluoro-, ion(1-) (9CI)",
                          "4,4,5,5,6,6,7,7,8,8,8-Undecafluorooctanoic acid",
                          "1-Hexanesulfonic acid, 1,1,2,2,3,3,4,4,5,5,6,6,6-tridecafluoro-",
                          "3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12-Heneicosafluoro-1-dodecanol",
                          "1-Heptanesulfonic acid, pentadecafluoro- (6CI)",
                          "1,1,2,2,3,3,4,4,5,5,6,6,6-Tridecafluoro-1-hexanesulfonamide",
                          "1-Butanesulfonic acid, 1,1,2,2,3,3,4,4,4-nonafluoro-, potassium salt (8CI,9CI)",
                          "1,1,2,2,3,3,4,4,4-Nonafluoro-N-methyl-1-butanesulfonamide",
                          "3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12-Henicosafluorododecane-1-sulfonic acid",
                          "Glycine, N-[(heptadecafluorooctyl)sulfonyl]-N-methyl- (9CI)", "C8 Cl-PFESA",
                          "Perfluoro compounds",
                          "Perfluorooctane sulfonamidoacetic acid", "Perfluoro compounds, γ-ω-perfluoro C10-20-thiols",
                          "Decanoic acid, nonadecafluoro-, ethenyl ester (9CI)",
                          "Octanoic acid, pentadecafluoro-, butyl ester (6CI,8CI,9CI)",
                          "Octanoic acid, tetradecafluoro-7-(trifluoromethyl)- (8CI)",
                          "Silane, trichloro[5,5,6,6,7,7,8,8,9,9,10,10,10-tridecafluoro-2-(1,1,2,2,3,3,4,4,5,5,6,6,6-tridecafluorohexyl)decyl]-",
                          "Undecane, 1,1,1,2,2,3,3,4,4,5,5,6,6-tridecafluoro-",
                          "1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13-Heptacosafluoro-15-iodopentadecane",
                          "1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10-Heneicosafluoro-12-iodooctadecane",
                          "1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8-Heptadecafluoro-10-iodohexadecane",
                          "1-Decanol, 3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-heptadecafluoro-, dihydrogen phosphate",
                          "1-Dodecanol, 3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12-heneicosafluoro-, hydrogen sulfate, ammonium salt",
                          "3,3,4,4,5,5,6,6,7,7,8,8,8-Tridecafluoro-1-octanol",
                          "3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-Heptadecafluoro-1-decanethiol",
                          "3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,14-Pentacosafluoro-1-tetradecanol",
                          "3,3,4,4,5,5,6,6,7,7,8,8,9,9,9-Pentadecafluoro-1-nonanol",
                          "4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-Pentadecafluoro-1-decanol",
                          "4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-Heptadecafluoroundecanenitrile",
                          "4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-Heptadecafluoroundecanoyl chloride",
                          "Heptadecane, 1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10-heneicosafluoro-12-iodo",
                          "Hexadecane, 1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10-heneicosafluoro-12-iodo",
                          "Dodecanoic acid, tricosafluoro-, ethyl ester",
                          "Dodecanoic acid, tricosafluoro-, methyl ester",
                          "Undecanal, heneicosafluoro",
                          "1-Octanesulfonic acid, 3,3,4,4,5,5,6,6,7,7,8,8,8-tridecafluoro-, ammonium salt (9CI)",
                          "1-Octanesulfonic acid, 3,3,4,4,5,5,6,6,7,7,8,8,8-tridecafluoro-, barium salt (2:1)",
                          "1-Octanesulfonic acid, 3,3,4,4,5,5,6,6,7,7,8,8,8-tridecafluoro-, potassium salt (9CI)",
                          "Glycine, N-[(heptadecafluorooctyl)sulfonyl]-N-methyl- (9CI)",
                          "Acrylic acid, 2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-nonadecafluorodecyl ester (8CI)"]

    # These are polyfluoroalkyl substances
    non_polymer_POLYs = ["2,2,3-Trifluoro-3-[1,1,2,2,3,3-hexafluoro-3-(trifluoromethoxy)propoxy]propanoic acid",
                         "FTSA", "FOSE", "MeFOSA", "FTOH", "DiPAP", "FTAB", "4:2 FTSA", "N-Et-FOSA", "N-Et-FOSA-A",
                         "ADONA", "GenX",
                         "N-EtFOSE", "N-MeFOSE", "FOSA", "FOSA_branched", "6:2 FTAC", "6:2 FTMAC", "6:2 FTSA",
                         "6:2 FTUCA", "PFODA", "PFOPA", "PFOSi", "PFPeS",
                         "6:2 PAP", "6:2FTS", "8:2 FTAC", "8:2 FTOH", "8:2 FTSA", "8:2 FTUCA",
                         "FBSE", "FBSA", "FTEO", "FTMAC", "FTS", "FTCA", "sulfonamido", "PolyFEA",
                         "FTUOH", "FTO", "FTUAL", "PAP", "FASE", "FASAA", "FAS(M)AC", "HFPO", "FTUCA", "FTCA",
                         "HFPO-DA", "PFBA", "10:2 FTUCA", "FTOH", "FTS", "PFDA",
                         "Acrylic acid, 2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9", "MeFBSE", "MeFBSAA", "n-MeFOSA",
                         "6:2 diPAP", "8:2 diPAP", "6:2/8:2 diPAP",
                         "Acrylic acid, 2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9-hexadecafluorononyl ester (7CI,8CI)",
                         "Acrylic acid, 3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,14-pentacosafluorotetradecyl ester (8CI)",
                         "Methacrylic acid, 2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-nonadecafluorodecyl ester (8CI)",
                         "Nonanoic acid, 4,4,5,5,6,6,7,7,8,8,9,9,9-tridecafluoro-2-iodo-, ethyl ester",
                         "Carbonic acid, ethenyl 3,3,4,4,5,5,6,6,7,7,8,8,8-tridecafluorooctyl ester",
                         "Oxirane, (2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-heneicosafluoroundecyl)- (9CI)",
                         "Oxirane, (2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,13-pentacosafluorotridecyl)- (9CI)",
                         "Oxirane, [2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,13,13,13-tetracosafluoro-12-(trifluoromethyl)tridecyl]- (9CI)",
                         "Ethanol, 2,2-difluoro-2-[1,1,2,2-tetrafluoro-2-[1,1,2,2-tetrafluoro-2-(nonafluorobutoxy)ethoxy]ethoxy]-",
                         "4-Bromo-2-[4,4,5,5,6,6,7,7,8,9,9,9-dodecafluoro-8-(trifluoromethyl)nonyl]phenol",
                         "2,2,3,4,4,5,5,6,6,7,8,8,8-Tridecafluoro-3,7-bis(trifluoromethyl)octanoic acid",
                         "2-Propanol, 1,3-bis[(2,2,3,3,4,4,5,5,6,6,7,7-dodecafluoroheptyl)oxy]-, hydrogen sulfate, sodium salt (9CI)",
                         "Phosphonic acid, P-(1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8-heptadecafluorooctyl)-, compd. with 4-methylbenzenamine (1:1)",
                         "Dichlorobis(3,3,4,4,5,5,6,6,6-nonafluorohexyl)silane",
                         "5-Fluoro-5,7,7-tris(trifluoromethyl)-6-[2,2,2-trifluoro-1-(trifluoromethyl)ethylidene]-1,4-dioxepane",
                         "Ethanol, 2,2-difluoro-2-[1,1,2,2-tetrafluoro-2-[1,1,2,2-tetrafluoro-2-(nonafluorobutoxy)ethoxy]ethoxy]- (9CI)",
                         "Ethanol, 2-[(2,2,3,3,4,4,5,5,6,6,7,7,7-tridecafluoroheptyl)oxy]-",
                         "1,2-Tridecanediol, 4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,13-heneicosafluoro-, 1-(dihydrogen phosphate), diammonium salt",
                         "1-Decanol, 4,4,5,5,6,6,7,7,8,8,9,9,10,10,10-pentadecafluoro-2-iodo",
                         "1-Dodecanol, 3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12-heneicosafluoro-, 1-(hydrogen sulfate), potassium salt",
                         "1-Propanol, 3-[(nonafluorobutyl)sulfonyl]",
                         "2,3,4,5,5,5-Hexafluoro-2,4-bis(trifluoromethyl)-1-pentanol",
                         "3,3,4,4,5,5,6,6,7,8,8,8-Dodecafluoro-7-(trifluoromethyl)-1-octanol",
                         "4,4,5,5,6,6,7,7,8,9,9,9-Dodecafluoro-2-iodo-8-(trifluoromethyl)-1-nonanol",
                         "4-[(4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-Heptadecafluoroundecyl)oxy]benzaldehyde",
                         "6,6,7,7,8,8,9,9,10,10,11,11,11-Tridecafluoro-4-iodo-1-undecanol",
                         "Octanal, pentadecafluoro", "2,2,3,3,4,4-Hexafluoro-4-(trifluoromethoxy)butanoic acid",
                         "Octane, 1,1,1,2,2,3,3,4,4,5,5,6,6-tridecafluoro-8-(2-propenyloxy)",
                         "Pentadecane, 1,1,1,2,2,3,3,4,4,5,5,6,6,7,7-pentadecafluoro-9-iodo",
                         "Pentanoic acid, 4,4-difluoro-5-[[(1,1,2,2,3,3,4,4,4-nonafluorobutyl)sulfonyl]oxy]-, ethyl ester",
                         "Phenol, 4-chloro-2-(2-chloro-4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-heptadecafluoroundecyl)-, acetate",
                         "Phenol, 5-chloro-2-(2-chloro-4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-heptadecafluoroundecyl)-, acetate",
                         "Potassium 2,2,3,3-tetrafluoro-3-(heptafluoropropoxy)propionate",
                         "Propanoic acid, 2,2,3,3-tetrafluoro-3-(heptafluoropropoxy)- (9CI)",
                         "Propanoic acid, 2,2,3,3-tetrafluoro-3-(heptafluoropropoxy)-, sodium salt (9CI)",
                         "Propanoic acid, 2,3,3,3-tetrafluoro-2-(heptafluoropropoxy)-, potassium salt (9CI)",
                         "Propanoic acid, 2,3,3,3-tetrafluoro-2-(heptafluoropropoxy)-, sodium salt (9CI)",
                         "Propanoic acid, 2,3,3,3-tetrafluoro-2-[1,1,2,3,3,3-hexafluoro-2-[1,1,2,3,3,3-hexafluoro-2-(heptafluoropropoxy)propoxy]propoxy]- (9CI)"
                         ]

    # Check if there are polymer PFAs - Large molecular weight long chained polymers.
    polymers = ["PTFE", "PVDF", "PVF", "PFPE", "FEP", "PFA", "PCTFE", "ETFE", "ECTFE", "FFPM", "FFKM",
                "FEPM", "PFSA", "PVDF", "HFP", "EFEP", "FFKM", "FEPM", "PCTFE", "Polychlorotrifluoroethylene",
                ]

    # Loop through df and check the string inside substance column.
    # Assign PFA Type to each row in df based on the substance type
    df["PFA type"] = ""
    for index, row in df.iterrows():
        substance = row["substance"]
        if substance in non_polymers_PERFs:
            df.at[index, "PFA type"] = "Perfluoroalkyl PFAs"
        elif substance in non_polymer_POLYs:
            df.at[index, "PFA type"] = "Polyfluoroalkyl PFAs"
        elif substance in polymers:
            df.at[index, "PFA type"] = "Polymer PFAs"
        else:
            df.at[index, "PFA type"] = "Unclassified"

    return df


preprocessed_and_flagged_final = add_pfa_group(preprocessed_and_flagged)

# Create a reduced columns version for faster loading in Dash
preprocessed_and_flagged_final_reduced = preprocessed_and_flagged_final.drop(columns=["sector", "measurement units",
                                                                                      "category", "date", "type"])

# Write out reduced version:
your_path = "path/file_name.parquet"
"""
preprocessed_and_flagged_final_reduced.to_parquet(
    your_path,
    index=False,
    engine="fastparquet")
"""
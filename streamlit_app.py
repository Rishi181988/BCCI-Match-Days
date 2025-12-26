import streamlit as st
import requests
import json
import re
import pandas as pd
import time
import os
from datetime import datetime
from collections import defaultdict
from io import BytesIO

from bcci_upcoming_matches_scraper import BCCIUpcomingMatchesScraper as RealScraper

st.title('🎈 BCCI Match Data')

st.write('This app compile all the BCCI domestic matches in one place')

# --- Scraper Class Integration ---
# (The provided scraper code is included here, with minor modifications
# to integrate with Streamlit's display and logging capabilities.)

def safe_strip(value):
    """Handle None values and clean whitespace"""
    return (value or 'N/A').strip()

class BCCIUpcomingMatchesScraper:
    def __init__(self, competitions_json: str | None = None, platforms: list[str] | None = None, genders: list[str] | None = None,
                 competition_ids: list[int] | None = None, competition_name_patterns: list[str] | None = None,
                 month: str | None = None, date_range: tuple[str, str] | None = None, scan_range: tuple[int, int] | None = None,
                 st_progress_bar=None, st_status_text=None): # Added Streamlit hooks
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.bcci.tv/"
        }
        # runtime configuration
        self.platforms = platforms or ["domestic"]
        self.genders = [g.lower() for g in (genders or ["men", "women"])]
        self.competitions_json = competitions_json
        # optional filters
        self.filter_comp_ids = set(competition_ids or [])
        self.filter_comp_name_patterns = [p.lower() for p in (competition_name_patterns or [])]
        self.filter_month = month  # 'YYYY-MM'
        self.filter_date_range = date_range  # (start_date_str, end_date_str)
        self.scan_range = scan_range  # (start_id, end_id)
        # competitions discovered at runtime: {int_id: {"name": str, "gender": str, "platform": str}}
        self.competitions: dict[int, dict] = {}
        self.upcoming_matches = []

        # Streamlit hooks
        self.st_progress_bar = st_progress_bar
        self.st_status_text = st_status_text
        self.total_comps = 0 # To track progress

    # --- Utility for Streamlit ---
    def _update_status(self, message: str, progress: float | None = None):
        """Updates the Streamlit status text and optional progress bar."""
        if self.st_status_text:
            self.st_status_text.text(message)
        # Only update progress if we have the total count
        if self.st_progress_bar and progress is not None and self.total_comps > 0:
            self.st_progress_bar.progress(min(1.0, progress))

    # -----------------------------
    # Competition discovery helpers (same as original, but using _update_status for feedback)
    # -----------------------------
    def discover_competitions_from_api(self, platform: str, filter_type: str) -> dict:
        """Discover competitions dynamically from BCCI 'getUpcomingMatches'."""
        url = f"https://scores2.bcci.tv/getUpcomingMatches?platform={platform}&filterType={filter_type}"
        self._update_status(f"Discovering competitions for {platform}/{filter_type}...")
        try:
            res = requests.get(url, headers=self.headers, timeout=20)
            res.raise_for_status()
            data = res.json()
            competitions: dict[int, dict] = {}
            # ... (rest of discovery logic is the same) ...
            for item in data.get("tournamentsAndSeries", []) or []:
                cid = item.get("competitionId")
                name = item.get("CompetitionName") or item.get("competitionName")
                if cid is None: continue
                try: cid_int = int(cid)
                except (TypeError, ValueError): continue
                competitions[cid_int] = {
                    "name": safe_strip(name),
                    "gender": "men" if filter_type.lower()=="men" else ("women" if filter_type.lower()=="women" else safe_strip(filter_type)),
                    "platform": safe_strip(platform)
                }
            for m in data.get("upcomingMatches", []) or []:
                cid = m.get("CompetitionID")
                name = m.get("CompetitionName")
                try: cid_int = int(cid)
                except (TypeError, ValueError): continue
                competitions.setdefault(cid_int, {
                    "name": safe_strip(name),
                    "gender": "men" if filter_type.lower()=="men" else ("women" if filter_type.lower()=="women" else safe_strip(filter_type)),
                    "platform": safe_strip(platform)
                })
            return competitions
        except Exception as e:
            self._update_status(f"Failed discovery for {platform}/{filter_type}: {e}")
            return {}

    def load_competitions_from_json(self, path: str) -> dict:
        """Load competitions mapping from a JSON file. (Simplified for Streamlit: disabled, as file upload is complex in this context)"""
        # In a real Streamlit app, you would handle st.file_uploader, but here we disable local file loading for simplicity.
        self._update_status(f"Note: Local JSON config loading is disabled in this web application.")
        return {}

    def build_competitions(self):
        """Build competitions dictionary via API across selected platforms and genders."""
        comps: dict[int, dict] = {}

        merged: dict[int, dict] = {}
        for platform in self.platforms:
            for gender in self.genders:
                part = self.discover_competitions_from_api(platform, gender)
                for cid, meta in part.items():
                    if cid not in merged:
                        merged[cid] = meta
                    else:
                        if not merged[cid].get("gender") or merged[cid].get("gender") == "Unknown":
                            merged[cid]["gender"] = meta.get("gender")
                        if not merged[cid].get("platform") or merged[cid].get("platform") == "unknown":
                            merged[cid]["platform"] = meta.get("platform")
        comps = merged

        # Optional scan range to broaden discovery
        if self.scan_range:
            start_id, end_id = self.scan_range
            try:
                self._update_status(f"Scanning competition IDs from {start_id} to {end_id}...")
                scanned = self._scan_competitions_range(int(start_id), int(end_id))
                for cid, meta in scanned.items():
                    comps.setdefault(cid, meta)
                if scanned:
                    self._update_status(f"Scan added {len(scanned)} competitions.")
            except Exception as e:
                self._update_status(f"Scan range error: {e}")

        self.competitions = comps
        if self.competitions:
            self._update_status(f"Discovered {len(self.competitions)} competitions.")
        else:
            self._update_status("Warning: No competitions discovered.")

    # -----------------------------
    # Optional range scanning for competitions (same as original)
    # -----------------------------
    def _try_fetch_competition_meta(self, comp_id: int) -> dict | None:
        """Attempt to fetch a competition feed and return minimal meta if valid, else None."""
        try:
            url = f"https://scores.bcci.tv/feeds/{comp_id}-matchschedule.js"
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code != 200: return None
            json_str = re.sub(r'^MatchSchedule\(|\);?\s*$', '', r.text, flags=re.DOTALL)
            data = json.loads(json_str)
            matches = data.get('Matchsummary') or []
            if not matches: return None
            name = safe_strip(matches[0].get('CompetitionName')) if matches else f"Competition {comp_id}"
            upper = name.upper()
            if 'WOMEN' in upper or "WOMEN'S" in upper: gender = 'women'
            else: gender = 'men'
            platform = (self.platforms[0] if self.platforms else 'domestic')
            return {"name": name, "gender": gender, "platform": platform}
        except Exception:
            return None

    def _scan_competitions_range(self, start_id: int, end_id: int) -> dict:
        """Scan a range of competition IDs and include those that return valid feeds."""
        found: dict[int, dict] = {}
        for cid in range(start_id, end_id + 1):
            if cid in self.competitions or cid in found: continue
            meta = self._try_fetch_competition_meta(cid)
            if meta:
                found[cid] = meta
                time.sleep(0.1) # Be polite during scan
        return found

    # -----------------------------
    # Match filtering and extraction (same as original)
    # -----------------------------
    def _parse_date(self, s: str):
        if not s: return None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
            try: return datetime.strptime(s, fmt).date()
            except Exception: continue
        return None

    def _passes_date_filters(self, date_str: str) -> bool:
        d = self._parse_date(date_str)
        if d is None: return True

        # Filter by month (not exposed in Streamlit UI, but kept for completeness)
        if self.filter_month:
            try:
                y, m = map(int, self.filter_month.split('-'))
                if not (y == d.year and m == d.month): return False
            except Exception: pass

        # Filter by date range (used by Streamlit UI)
        if self.filter_date_range:
            try:
                start_s, end_s = self.filter_date_range
                start_d = datetime.strptime(start_s, "%Y-%m-%d").date()
                end_d = datetime.strptime(end_s, "%Y-%m-%d").date()
                if not (start_d <= d <= end_d): return False
            except Exception: pass
        return True

    def should_include_match(self, match_data) -> bool:
        """Apply date/month filters."""
        date_str = match_data.get('MatchDate') or match_data.get('matchDate') or ''
        return self._passes_date_filters(date_str)

    def extract_match_data(self, match, comp_id, comp_name, gender, platform):
        """Extract the required fields from match data (same as original)."""
        status = match.get('matchStatus') or match.get('MatchStatus') or ''
        status_norm = safe_strip(status).lower().capitalize() if status else 'N/A'
        return {
            'Platform': 'Domestic' if str(platform).lower()=='domestic' else safe_strip(platform).capitalize(),
            'CompetitionID': comp_id,
            'CompetitionName': safe_strip(match.get('CompetitionName') or comp_name),
            'CompetitionGender': 'Men' if str(gender).lower()=='men' else ('Women' if str(gender).lower()=='women' else safe_strip(gender)),
            'MatchStatus': status_norm,
            'MatchOrder': safe_strip(match.get('MatchOrder')),
            'MatchID': safe_strip(match.get('MatchID')),
            'MatchTypeID': safe_strip(match.get('MatchTypeID')),
            'MatchType': safe_strip(match.get('MatchType')),
            'MatchName': safe_strip(match.get('MatchName')),
            'MatchDate': safe_strip(match.get('MatchDate')),
            'GroundUmpire1': safe_strip(match.get('GroundUmpire1')),
            'GroundUmpire2': safe_strip(match.get('GroundUmpire2')),
            'ThirdUmpire': safe_strip(match.get('ThirdUmpire')),
            'Referee': safe_strip(match.get('Referee')),
            'GroundName': safe_strip(match.get('GroundName'))
        }

    def fetch_competition_data(self, comp_id, current_comp_index):
        """Fetch match data for a specific competition, updating Streamlit status."""
        try:
            meta = self.competitions.get(comp_id, {"name": f"Competition {comp_id}", "gender": "Unknown", "platform": "unknown"})
            default_name = meta.get("name", f"Competition {comp_id}")
            gender = meta.get("gender", "Unknown")
            platform = meta.get("platform", "unknown")
            url = f"https://scores.bcci.tv/feeds/{comp_id}-matchschedule.js"

            progress = (current_comp_index + 1) / self.total_comps
            self._update_status(f"({current_comp_index + 1}/{self.total_comps}) Fetching: {default_name} (ID: {comp_id})", progress)

            response = requests.get(url, headers=self.headers, timeout=15)

            if response.status_code != 200:
                self._update_status(f"Failed to fetch data for competition {comp_id}: HTTP {response.status_code}")
                return 0

            # Clean JSONP response
            json_str = re.sub(r'^MatchSchedule\(|\);?\s*$', '', response.text, flags=re.DOTALL)

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                self._update_status(f"Failed to parse JSON for competition {comp_id}: {e}")
                return 0

            matches_processed = 0
            for match in data.get('Matchsummary', []) or []:
                if self.should_include_match(match):
                    comp_name_from_match = safe_strip(match.get('CompetitionName'))
                    comp_name = comp_name_from_match if comp_name_from_match != 'N/A' else default_name
                    match_data = self.extract_match_data(match, comp_id, comp_name, gender, platform)
                    self.upcoming_matches.append(match_data)
                    matches_processed += 1

            return matches_processed

        except Exception as e:
            self._update_status(f"Error processing competition {comp_id}: {str(e)}")
            return 0

    def scrape_all_competitions(self):
        """Scrape matches from all discovered competitions with optional filters"""
        self.upcoming_matches = []

        # 1. Build the competitions map dynamically
        self._update_status("Starting competition discovery...")
        self.build_competitions()

        # 2. Apply competition ID/name filters
        comp_items = list(self.competitions.items())
        if self.filter_comp_ids:
            comp_items = [(cid, meta) for cid, meta in comp_items if cid in self.filter_comp_ids]
        if self.filter_comp_name_patterns:
            patterns = self.filter_comp_name_patterns
            comp_items = [
                (cid, meta) for cid, meta in comp_items
                if any(p in (meta.get('name','').lower()) for p in patterns)
            ]

        self.total_comps = len(comp_items)
        if self.total_comps == 0:
            self._update_status("No competitions matched your filters. Scraping aborted.")
            return 0

        # 3. Fetch data for filtered competitions
        total_matches = 0
        for i, (comp_id, _meta) in enumerate(comp_items):
            matches_count = self.fetch_competition_data(comp_id, i)
            total_matches += matches_count
            time.sleep(1.0) # Add delay to be respectful

        self._update_status(f"Scraping complete! Found {total_matches} total matches.", 1.0)
        return total_matches

    def get_dataframe(self) -> pd.DataFrame:
        """Converts scraped data to a DataFrame, applies sorting and standard formatting."""
        if not self.upcoming_matches:
            return pd.DataFrame()

        df = pd.DataFrame(self.upcoming_matches)

        # Reorder columns
        desired_order = [
            'Platform', 'CompetitionGender', 'CompetitionID', 'CompetitionName', 'MatchStatus',
            'MatchOrder', 'MatchID', 'MatchTypeID', 'MatchType', 'MatchName', 'MatchDate',
            'GroundUmpire1', 'GroundUmpire2', 'ThirdUmpire', 'Referee', 'GroundName'
        ]
        cols = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
        df = df[cols]

        # Sort by Platform, CompetitionGender, CompetitionID and MatchDate
        sort_cols = [c for c in ['Platform', 'CompetitionGender', 'CompetitionID', 'MatchDate'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=[True, True, True, False]) # Sort date descending for recency

        return df

    def to_excel_bytes(self, df: pd.DataFrame) -> bytes:
        """Exports the DataFrame to an in-memory Excel file (bytes)."""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='BCCI Matches', index=False)

            # --- Apply formatting (similar to original scraper) ---
            workbook = writer.book
            worksheet = writer.sheets['BCCI Matches']

            header_format = workbook.add_format({'bold': True, 'bg_color': '#E0E0FF', 'valign': 'vcenter', 'align': 'center', 'text_wrap': False, 'border': 1})
            data_format = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'border': 1})

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            column_widths = {
                'Platform': 12, 'CompetitionGender': 12, 'CompetitionID': 12, 'CompetitionName': 28, 'MatchStatus': 12,
                'MatchOrder': 12, 'MatchID': 12, 'MatchTypeID': 12, 'MatchType': 22, 'MatchName': 34, 'MatchDate': 14,
                'GroundUmpire1': 22, 'GroundUmpire2': 22, 'ThirdUmpire': 20, 'Referee': 20, 'GroundName': 28
            }

            for col_num, column in enumerate(df.columns):
                width = column_widths.get(column, 15)
                # Set column width
                worksheet.set_column(col_num, col_num, width)
                # Apply data format (explicitly applying data_format to cells is better than conditional_format here)
                # Note: Conditional format is powerful but `set_column` already sets default cell format,
                # for simplicity and Streamlit compatibility, we rely on the default format after set_column,
                # but we will manually apply to header.
            # No need for conditional format if we rely on set_column's default format.

        return output.getvalue()

# --- Streamlit Application Main Logic ---

def parse_multi_input(input_str: str, target_type=str) -> list:
    """Parses a comma-separated string into a list of specified type."""
    if not input_str:
        return []
    items = [x.strip() for x in input_str.split(',') if x.strip()]
    if target_type == int:
        return [int(item) for item in items if item.isdigit()]
    return items

def parse_date_range(date_tuple) -> tuple[str, str] | None:
    """Converts a (datetime.date, datetime.date) tuple to (YYYY-MM-DD, YYYY-MM-DD) strings."""
    if date_tuple and len(date_tuple) == 2:
        return (date_tuple[0].strftime("%Y-%m-%d"), date_tuple[1].strftime("%Y-%m-%d"))
    return None

def parse_scan_range(scan_str: str) -> tuple[int, int] | None:
    """Parses '300-400' into (300, 400)."""
    if not scan_str or '-' not in scan_str:
        return None
    try:
        a, b = scan_str.split('-')
        start = int(a.strip())
        end = int(b.strip())
        if start > 0 and end >= start:
            return (start, end)
    except Exception:
        pass
    return None

def app():
    st.set_page_config(layout="wide", page_title="BCCI Match Data Scraper")

    st.title("🏏 BCCI Match Data Scraper")
    st.markdown("Use the filters in the sidebar to configure the scrape and retrieve match data (All statuses: Upcoming, Live, Finished).")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration Filters")

    # Platforms
    platforms = st.sidebar.multiselect(
        "Platforms",
        options=["domestic", "international"],
        default=["domestic"]
    )

    # Genders
    genders = st.sidebar.multiselect(
        "Genders",
        options=["men", "women"],
        default=["men", "women"]
    )

    # Competition Filters
    st.sidebar.subheader("Competition Filters")
    comp_ids_str = st.sidebar.text_input(
        "Competition IDs (comma-separated)",
        placeholder="e.g., 316, 317, 318"
    )
    comp_names_str = st.sidebar.text_input(
        "Competition Name Search (comma-separated substrings)",
        placeholder="e.g., ranji, irani, vijay"
    )

    # Date Filters
    st.sidebar.subheader("Date Filters (Match Date)")
    date_range_tuple = st.sidebar.date_input(
        "Select Date Range (Optional)",
        value=[], # Start with empty range
        min_value=datetime(2020, 1, 1).date(),
        max_value=datetime.now().date().replace(year=datetime.now().year + 5)
    )

    # Advanced Scan
    st.sidebar.subheader("Advanced Discovery")
    scan_range_str = st.sidebar.text_input(
        "Competition ID Scan Range (e.g., 300-400)",
        help="Scans a range of IDs to discover potentially unlisted competitions. Slows down the process."
    )

    # --- Main Area ---
    st.markdown("---")

    # Status placeholders
    status_container = st.empty()
    progress_bar = st.progress(0)

    # Button to start scraping
    if st.button("🚀 Start Scraping", type="primary"):

        # 1. Parse Inputs
        comp_ids = parse_multi_input(comp_ids_str, int)
        comp_names = parse_multi_input(comp_names_str, str)
        date_range = parse_date_range(date_range_tuple)
        scan_range = parse_scan_range(scan_range_str)

        try:
            # 2. Initialize Scraper
            scraper = BCCIUpcomingMatchesScraper(
                platforms=platforms,
                genders=genders,
                competition_ids=comp_ids,
                competition_name_patterns=comp_names,
                date_range=date_range,
                scan_range=scan_range,
                st_progress_bar=progress_bar,
                st_status_text=status_container
            )

            # 3. Execute Scraping
            status_container.info("Starting data retrieval...")
            total_matches = scraper.scrape_all_competitions()

            # 4. Process Results
            df = scraper.get_dataframe()

            progress_bar.progress(1.0)
            status_container.success(f"✅ Scraping finished! Found {total_matches} total matches.")

            if not df.empty:
                st.subheader(f"Results ({len(df)} Matches)")
                st.dataframe(df, use_container_width=True)

                # 5. Download Button
                excel_bytes = scraper.to_excel_bytes(df)

                st.download_button(
                    label="💾 Download Data as Excel (XLSX)",
                    data=excel_bytes,
                    file_name="bcci_match_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Click to download the full dataset with formatting."
                )
            else:
                st.warning("⚠️ No matches were found based on the provided filters.")

        except requests.exceptions.Timeout:
            status_container.error("Network Timeout Error: The request took too long to complete. Please try again.")
        except requests.exceptions.RequestException as e:
            status_container.error(f"A Network Error Occurred: {e}")
        except Exception as e:
            status_container.error(f"An unexpected error occurred: {e}")

        # Reset progress bar state after error or completion
        progress_bar.empty()


# ===============================
# Dashboard mode (analyze Excel)
# ===============================
RAW_SHEET = "Upcoming Matches"
UMPIRE_SHEET = "Umpire Days"
REFEREE_SHEET = "Referee Days"
EXCEL_FILE = "bcci_all_competitions.xlsx"
REQUIRED_COLUMNS = [
    'Platform', 'CompetitionGender', 'CompetitionID', 'CompetitionName', 'MatchStatus',
    'MatchOrder', 'MatchID', 'MatchTypeID', 'MatchType', 'Number of Days', 'MatchName', 'MatchDate',
    'GroundUmpire1', 'GroundUmpire2', 'ThirdUmpire', 'Referee', 'GroundName'
]

@st.cache_data(show_spinner=False)
def _load_excel_cached(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found at '{path}'. Please run the scraper to generate it.")
    try:
        dfs = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}")
    for sheet in (RAW_SHEET, UMPIRE_SHEET, REFEREE_SHEET):
        if sheet not in dfs:
            raise KeyError(f"Missing sheet '{sheet}'. Found sheets: {list(dfs.keys())}")
    raw = dfs[RAW_SHEET].copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"The following required columns are missing in '{RAW_SHEET}': {missing}")
    # typing
    def _to_date(s):
        if pd.isna(s) or s == '':
            return None
        try:
            return pd.to_datetime(s, errors='coerce').date()
        except Exception:
            return None
    raw['MatchDate'] = raw['MatchDate'].apply(_to_date)
    raw['Number of Days'] = pd.to_numeric(raw['Number of Days'], errors='coerce').fillna(0.0)
    return {RAW_SHEET: raw, UMPIRE_SHEET: dfs[UMPIRE_SHEET].copy(), REFEREE_SHEET: dfs[REFEREE_SHEET].copy()}

def _pivot_umpires(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Number of Days'] = pd.to_numeric(df['Number of Days'], errors='coerce').fillna(0.0)
    base = ['CompetitionName', 'Number of Days']
    u1 = df[base + ['GroundUmpire1']].rename(columns={'GroundUmpire1': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
    u2 = df[base + ['GroundUmpire2']].rename(columns={'GroundUmpire2': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
    u3 = df[base + ['ThirdUmpire']].rename(columns={'ThirdUmpire': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
    umps = pd.concat([u1, u2, u3], ignore_index=True)
    umps['Umpire Name'] = umps['Umpire Name'].fillna('').astype(str).str.strip()
    umps = umps[umps['Umpire Name'] != '']
    comps = sorted([c for c in umps['Tournament Name'].dropna().unique().tolist() if c])
    pv = umps.pivot_table(index='Umpire Name', columns='Tournament Name', values='Number of Days', aggfunc='sum', fill_value=0.0)
    pv = pv.reindex(columns=comps, fill_value=0.0)
    pv['Total'] = pv.sum(axis=1)
    # Sort with N/A entries at the bottom
    pv = pv.reset_index()
    pv['is_na'] = pv['Umpire Name'].str.upper().isin(['N/A', 'NA'])
    pv = pv.sort_values(['is_na', 'Total'], ascending=[True, False])
    pv = pv.drop(columns=['is_na'])
    return pv.reset_index(drop=True)

def _pivot_referees(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Number of Days'] = pd.to_numeric(df['Number of Days'], errors='coerce').fillna(0.0)
    ref = df[['CompetitionName', 'Number of Days', 'Referee']].rename(columns={'Referee': 'Referee Name', 'CompetitionName': 'Tournament Name'})
    ref['Referee Name'] = ref['Referee Name'].fillna('').astype(str).str.strip()
    ref = ref[ref['Referee Name'] != '']
    comps = sorted([c for c in ref['Tournament Name'].dropna().unique().tolist() if c])
    pv = ref.pivot_table(index='Referee Name', columns='Tournament Name', values='Number of Days', aggfunc='sum', fill_value=0.0)
    pv = pv.reindex(columns=comps, fill_value=0.0)
    pv['Total'] = pv.sum(axis=1)
    # Sort with N/A entries at the bottom
    pv = pv.reset_index()
    pv['is_na'] = pv['Referee Name'].str.upper().isin(['N/A', 'NA'])
    pv = pv.sort_values(['is_na', 'Total'], ascending=[True, False])
    pv = pv.drop(columns=['is_na'])
    return pv.reset_index(drop=True)

def _write_formatted_excel(buf: BytesIO, df_main: pd.DataFrame, ump_pivot: pd.DataFrame, ref_pivot: pd.DataFrame):
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_main.to_excel(writer, sheet_name=RAW_SHEET, index=False)
        ump_pivot.to_excel(writer, sheet_name=UMPIRE_SHEET, index=False)
        ref_pivot.to_excel(writer, sheet_name=REFEREE_SHEET, index=False)
        workbook = writer.book
        header_format = workbook.add_format({'bold': True,'bg_color': '#FFFF00','valign':'vcenter','align':'center','text_wrap': False})
        data_format = workbook.add_format({'valign':'vcenter','align':'center','right':1,'num_format':'0.0'})
        bottom_border = workbook.add_format({'bottom': 2})
        right_border = workbook.add_format({'right': 2})
        left_border  = workbook.add_format({'left':  2})

        def fmt(ws, df_local, is_pivot_sheet=False):
            ws.set_column(0, 0, 15, workbook.add_format({'align':'left','valign':'vcenter'}))
            ws.set_column(1, max(1, len(df_local.columns)-1), 10, data_format)
            for col_num, value in enumerate(df_local.columns.values):
                ws.write(0, col_num, value, header_format)
                ws.conditional_format(0, col_num, 0, col_num, {'type':'no_blanks','format': workbook.add_format({'bold':True,'bg_color':'#FFFF00','valign':'vcenter','align':'center','top':2,'bottom':2,'left':2,'right':2,'text_wrap':False})})
            max_row = len(df_local); max_col = max(0, len(df_local.columns)-1)
            ws.conditional_format(1,0,max_row,max_col,{'type':'no_blanks','format': right_border})
            ws.conditional_format(1,0,max_row,max_col,{'type':'no_blanks','format': left_border})
            ws.conditional_format(max_row,0,max_row,max_col,{'type':'no_blanks','format': bottom_border})
            ws.hide_gridlines(2)
            ws.set_default_row(hide_unused_rows=True)
            ws.set_selection(max_row+1,0,max_row+1,max_col+1)
            ws.set_row(max_row+2, None, None, {'hidden': True})
            ws.set_column(max_col+2, 16383, None, None, {'hidden': True})

            # Special formatting for pivot sheets (Umpire/Referee workload)
            if is_pivot_sheet and len(df_local.columns) > 1:
                # Auto-fit the first column (official name) based on content
                max_name_length = max(
                    df_local.iloc[:, 0].astype(str).str.len().max() if len(df_local) > 0 else 10,
                    len(str(df_local.columns[0]))
                )
                # Add some padding and set a reasonable max width
                col_width = min(max_name_length + 2, 50)
                ws.set_column(0, 0, col_width, workbook.add_format({'align':'left','valign':'vcenter'}))

                # Hide all middle columns (keep only first and last "Total" column visible)
                if len(df_local.columns) > 2:
                    # Hide columns from index 1 to second-to-last (max_col - 1)
                    ws.set_column(1, max_col - 1, None, None, {'hidden': True})

        fmt(writer.sheets[RAW_SHEET], df_main, is_pivot_sheet=False)
        fmt(writer.sheets[UMPIRE_SHEET], ump_pivot, is_pivot_sheet=True)
        fmt(writer.sheets[REFEREE_SHEET], ref_pivot, is_pivot_sheet=True)

def dashboard_main():
    st.title("BCCI Cricket Match Data Dashboard")
    st.write("Analyze BCCI cricket match data from the generated Excel file. Use the filters in the sidebar and download a formatted Excel of your filtered data.")
    st.markdown("""
    <style>
    .note { color: #666; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Loading Excel data..."):
            dfs = _load_excel_cached(EXCEL_FILE)
    except Exception as e:
        st.error(f"Error: {e}\n\nEnsure '{EXCEL_FILE}' exists and is closed, then refresh.")
        return
    raw = dfs[RAW_SHEET]

    # Helper function to categorize matches as Knockout or League
    def categorize_match(match_order):
        if pd.isna(match_order):
            return 'League Matches'
        match_order_lower = str(match_order).lower()
        knockout_keywords = ['final', 'semi', 'quarter', 'playoff', 'eliminator']
        if any(keyword in match_order_lower for keyword in knockout_keywords):
            return 'Knockout Matches'
        return 'League Matches'

    # Add match category column to raw data
    raw['MatchCategory'] = raw['MatchOrder'].apply(categorize_match) if 'MatchOrder' in raw.columns else 'League Matches'

    # Sidebar filters
    st.sidebar.header("Filters")

    # Get all unique values for initial options
    g_all = sorted(raw['CompetitionGender'].dropna().unique().tolist()) if 'CompetitionGender' in raw.columns else []
    dates = raw['MatchDate'].dropna()
    if not dates.empty:
        dmin = dates.min()
        dmax = dates.max()
    else:
        dmin = datetime.now().date()
        dmax = dmin

    if 'dash_filters_init' not in st.session_state:
        st.session_state['sel_g'] = g_all
        st.session_state['sel_cat'] = []
        st.session_state['sel_comp'] = []
        st.session_state['sel_type'] = []
        st.session_state['sel_order'] = []
        st.session_state['sel_range'] = (dmin, dmax)
        st.session_state['dash_filters_init'] = True

    # Validate that session state date range is within current data bounds
    if 'sel_range' in st.session_state and all(st.session_state['sel_range']):
        stored_min, stored_max = st.session_state['sel_range']
        if dmin and dmax and (stored_min < dmin or stored_max > dmax):
            st.session_state['sel_range'] = (dmin, dmax)

    if st.sidebar.button("Clear All Filters"):
        st.session_state['sel_g'] = g_all
        st.session_state['sel_cat'] = []
        st.session_state['sel_comp'] = []
        st.session_state['sel_type'] = []
        st.session_state['sel_order'] = []
        st.session_state['sel_range'] = (dmin, dmax)

    # FILTER 1: Gender (top of hierarchy)
    sel_g = st.sidebar.multiselect("Gender", options=g_all, default=st.session_state['sel_g'])
    st.session_state['sel_g'] = sel_g

    # Filter data by gender for cascading
    df_filtered_by_gender = raw.copy()
    if sel_g and 'CompetitionGender' in df_filtered_by_gender:
        df_filtered_by_gender = df_filtered_by_gender[df_filtered_by_gender['CompetitionGender'].isin(sel_g)]

    # FILTER 2: Match Category (depends on Gender)
    cat_options = sorted(df_filtered_by_gender['MatchCategory'].dropna().unique().tolist()) if 'MatchCategory' in df_filtered_by_gender.columns else []
    # Ensure consistent ordering: Knockout first, then League
    cat_options_ordered = []
    if 'Knockout Matches' in cat_options:
        cat_options_ordered.append('Knockout Matches')
    if 'League Matches' in cat_options:
        cat_options_ordered.append('League Matches')

    sel_cat = st.sidebar.multiselect("Match Category", options=cat_options_ordered, default=st.session_state['sel_cat'])
    st.session_state['sel_cat'] = sel_cat

    # Filter data by gender + category for cascading
    df_filtered_by_cat = df_filtered_by_gender.copy()
    if sel_cat and 'MatchCategory' in df_filtered_by_cat:
        df_filtered_by_cat = df_filtered_by_cat[df_filtered_by_cat['MatchCategory'].isin(sel_cat)]

    # FILTER 3: Competition Name (depends on Gender + Category)
    comps_options = sorted(df_filtered_by_cat['CompetitionName'].dropna().unique().tolist()) if 'CompetitionName' in df_filtered_by_cat.columns else []
    comps = st.sidebar.multiselect("Competition Name", options=comps_options, default=st.session_state['sel_comp'], help="Filter by competition")
    st.session_state['sel_comp'] = comps

    # Filter data by gender + category + competition for cascading
    df_filtered_by_comp = df_filtered_by_cat.copy()
    if comps and 'CompetitionName' in df_filtered_by_comp:
        df_filtered_by_comp = df_filtered_by_comp[df_filtered_by_comp['CompetitionName'].isin(comps)]

    # FILTER 4: Match Type (depends on Gender + Category + Competition)
    types_options = sorted(df_filtered_by_comp['MatchType'].dropna().unique().tolist()) if 'MatchType' in df_filtered_by_comp.columns else []
    mtypes = st.sidebar.multiselect("Match Type", options=types_options, default=st.session_state['sel_type'], help="Filter by match type (T20, One Day, Multi Day)")
    st.session_state['sel_type'] = mtypes

    # Filter data by gender + category + competition + match type for cascading
    df_filtered_by_mt = df_filtered_by_comp.copy()
    if mtypes and 'MatchType' in df_filtered_by_mt:
        df_filtered_by_mt = df_filtered_by_mt[df_filtered_by_mt['MatchType'].isin(mtypes)]

    # FILTER 5: Match Order (depends on all previous filters)
    orders_options = sorted(df_filtered_by_mt['MatchOrder'].dropna().unique().tolist()) if 'MatchOrder' in df_filtered_by_mt.columns else []
    morder = st.sidebar.multiselect("Match Order", options=orders_options, default=st.session_state['sel_order'], help="Stage or order (Final, Group Match, etc.)")
    st.session_state['sel_order'] = morder

    # FILTER 6: Date Range (independent)
    dr = st.sidebar.date_input("Date Range", value=st.session_state['sel_range'] if all(st.session_state['sel_range']) else None, min_value=dmin, max_value=dmax)
    if isinstance(dr, tuple) and len(dr)==2:
        st.session_state['sel_range'] = dr

    # Apply all filters to get final filtered data
    with st.spinner("Applying filters..."):
        df = raw.copy()
        if sel_g and 'CompetitionGender' in df:
            df = df[df['CompetitionGender'].isin(sel_g)]
        if sel_cat and 'MatchCategory' in df:
            df = df[df['MatchCategory'].isin(sel_cat)]
        if comps and 'CompetitionName' in df:
            df = df[df['CompetitionName'].isin(comps)]
        if mtypes and 'MatchType' in df:
            df = df[df['MatchType'].isin(mtypes)]
        if morder and 'MatchOrder' in df:
            df = df[df['MatchOrder'].isin(morder)]
        if all(st.session_state['sel_range']):
            start, end = st.session_state['sel_range']
            df = df[(df['MatchDate'] >= start) & (df['MatchDate'] <= end)]

        # Remove the temporary MatchCategory column from display
        if 'MatchCategory' in df.columns:
            df = df.drop(columns=['MatchCategory'])
    total_matches = len(raw)
    filtered_matches = len(df)
    st.caption(f"Filter Summary: Showing {filtered_matches} of {total_matches} matches")
    # Metrics
    st.sidebar.caption(f"Showing {filtered_matches} of {total_matches} matches")

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{filtered_matches}")
    c2.metric("Competitions", f"{df['CompetitionName'].nunique()}")
    c3.metric("Umpires (unique)", f"{pd.unique(pd.concat([df['GroundUmpire1'], df['GroundUmpire2']]).dropna()).size}")
    # Quick search across key text fields
    search = st.text_input("\U0001F50E Search", value="", help="Search across MatchName, GroundName, and officials")
    if search:
        q = str(search).lower()
        def _row_has_q(row):
            fields = [row.get('MatchName',''), row.get('GroundName',''), row.get('GroundUmpire1',''), row.get('GroundUmpire2',''), row.get('Referee','')]
            return any(q in str(x).lower() for x in fields)
        if not df.empty:
            df = df[df.apply(_row_has_q, axis=1)]

    # Raw table
    st.subheader("\U0001F4CA Match Details")
    if df.empty:
        st.warning("No data after filters. Adjust filters.")
    else:
        st.write(f"Rows: {len(df)}")
        st.dataframe(df, use_container_width=True, hide_index=True)
    # Umpire pivot
    st.subheader("\U0001F468\u200d\u2696\uFE0F Umpire Workload Analysis")
    with st.spinner("Computing umpire pivot..."):
        ump_pivot = _pivot_umpires(df)
    st.dataframe(ump_pivot, use_container_width=True, hide_index=True)
    # Referee pivot
    st.subheader("\U0001F3CF Referee Workload Analysis")
    with st.spinner("Computing referee pivot..."):
        ref_pivot = _pivot_referees(df)
    st.dataframe(ref_pivot, use_container_width=True, hide_index=True)
    # Download button
    st.subheader("\U0001F4E5 Download Filtered Data")
    if df.empty:
        st.caption("Nothing to download  no rows after filters.")
    else:
        out = BytesIO()
        _write_formatted_excel(out, df, ump_pivot, ref_pivot)
        out.seek(0)
        st.download_button(
            label="Download Excel",
            data=out,
            file_name="bcci_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Downloads three sheets with formatting (headers, borders, widths)"
        )

# ===============================
# Real-time scraping Streamlit UI
# ===============================
@st.cache_data(show_spinner=False)
def _cached_competitions() -> dict:
    s = RealScraper(platforms=["domestic"], genders=["men","women"])
    comps = s.discover_competitions_from_competition_js()
    if not comps:
        merged = {}
        for gender in ("men","women"):
            merged.update(s.discover_competitions_from_api("domestic", gender))
        comps = merged
    return comps or {}

def _format_comp_label(cid: int, comps: dict) -> str:
    meta = comps.get(cid, {})
    return f"{meta.get('name','Competition')} (ID: {cid})"

def _to_date_any(x):
    if pd.isna(x) or x=="":
        return None
    try:
        return pd.to_datetime(x, errors='coerce').date()
    except Exception:
        return None

def realtime_scraper_app():
    st.set_page_config(layout="wide", page_title="BCCI Real-time Scraper")
    st.title("🏏 BCCI Real-time Scraping Dashboard")
    st.caption("Discover competitions, scrape live from BCCI feeds, filter and analyze, and download formatted Excel.")

    # Phase 1: Discover competitions
    with st.spinner("Discovering competitions..."):
        comps = _cached_competitions()
    if not comps:
        st.error("No competitions discovered. Please retry later.")
        return
    comp_ids_sorted = sorted(comps.keys())

    # Sidebar selection
    st.sidebar.header("Competition Selection")
    select_all = st.sidebar.checkbox("Select All Competitions", value=True)
    options = comp_ids_sorted

    # When "Select All" is checked, show placeholder instead of all competition names
    if select_all:
        default = options
        # Use a custom placeholder by showing empty list in multiselect but tracking all in selected
        selected = st.sidebar.multiselect(
            "Competitions",
            options=options,
            default=[],  # Show empty to avoid clutter
            format_func=lambda cid: _format_comp_label(cid, comps),
            help=f"All {len(options)} competitions selected",
            disabled=True  # Disable when "Select All" is active
        )
        # Override with all options when select_all is True
        selected = options
        st.sidebar.caption(f"✅ All {len(options)} competitions selected")
    else:
        # When "Select All" is unchecked, allow manual selection
        selected = st.sidebar.multiselect(
            "Competitions",
            options=options,
            default=[],
            format_func=lambda cid: _format_comp_label(cid, comps),
            help="Select competitions to scrape"
        )

    # Phase 2: Scraping interface
    st.sidebar.header("Scraping")
    start_btn = st.sidebar.button("🚀 Start Scraping", type="primary")
    status = st.empty(); prog = st.progress(0)
    if start_btn:
        scraper = RealScraper(platforms=["domestic"], genders=["men","women"])
        scraper.competitions = {cid: comps[cid] for cid in selected} if selected else {cid: comps[cid] for cid in comp_ids_sorted}
        total = len(scraper.competitions)
        matches_total = 0
        for i, cid in enumerate(scraper.competitions.keys()):
            name = scraper.competitions[cid].get('name', f'Competition {cid}')
            status.info(f"({i+1}/{total}) Fetching: {name} (ID: {cid})")
            try:
                matches_total += scraper.fetch_competition_data(cid)
            except Exception as e:
                st.warning(f"Failed {cid}: {e}")
            prog.progress((i+1)/max(1,total))
            time.sleep(0.6)
        status.success(f"Done. Competitions processed: {total}. Matches found: {matches_total}.")

        df = pd.DataFrame(scraper.upcoming_matches)
        if not df.empty:
            desired = ['Platform','CompetitionGender','CompetitionID','CompetitionName','MatchStatus','MatchOrder','MatchID','MatchTypeID','MatchType','Number of Days','MatchName','MatchDate','GroundUmpire1','GroundUmpire2','ThirdUmpire','Referee','GroundName']
            cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
            if cols:
                df = df[cols]
            df['MatchDate'] = df['MatchDate'].apply(_to_date_any)
            df['Number of Days'] = pd.to_numeric(df.get('Number of Days', 0), errors='coerce').fillna(0.0)
        st.session_state['raw_df'] = df

    # Phase 3: Filters & Analysis
    df_raw = st.session_state.get('raw_df')
    if df_raw is None or df_raw.empty:
        st.info("Scrape some data to enable filtering and analysis.")
        return

    st.sidebar.header("Filters")

    # Helper function to categorize matches as Knockout or League
    def categorize_match(match_order):
        if pd.isna(match_order):
            return 'League Matches'
        match_order_lower = str(match_order).lower()
        knockout_keywords = ['final', 'semi', 'quarter', 'playoff', 'eliminator']
        if any(keyword in match_order_lower for keyword in knockout_keywords):
            return 'Knockout Matches'
        return 'League Matches'

    # Add match category column to raw data
    df_raw['MatchCategory'] = df_raw['MatchOrder'].apply(categorize_match) if 'MatchOrder' in df_raw.columns else 'League Matches'

    # Get all unique values for initial options
    g_all = sorted(df_raw['CompetitionGender'].dropna().unique().tolist()) if 'CompetitionGender' in df_raw.columns else []
    dates = df_raw['MatchDate'].dropna() if 'MatchDate' in df_raw.columns else pd.Series([], dtype='datetime64[ns]')
    dmin = dates.min() if not dates.empty else None
    dmax = dates.max() if not dates.empty else None

    # Initialize session state for filters
    if 'filters_init' not in st.session_state:
        st.session_state['f_g'] = g_all
        st.session_state['f_cat'] = []
        st.session_state['f_mt'] = []
        st.session_state['f_mo'] = []
        st.session_state['f_range'] = (dmin, dmax)
        st.session_state['filters_init'] = True

    # Validate that session state date range is within current data bounds
    if 'f_range' in st.session_state and all(st.session_state['f_range']):
        stored_min, stored_max = st.session_state['f_range']
        if dmin and dmax and (stored_min < dmin or stored_max > dmax):
            st.session_state['f_range'] = (dmin, dmax)

    # Clear Filters button
    if st.sidebar.button("Clear Filters"):
        st.session_state['f_g'] = g_all
        st.session_state['f_cat'] = []
        st.session_state['f_mt'] = []
        st.session_state['f_mo'] = []
        st.session_state['f_range'] = (dmin, dmax)

    # FILTER 1: Gender (top of hierarchy)
    sel_g = st.sidebar.multiselect("Gender", options=g_all, default=st.session_state['f_g'])
    st.session_state['f_g'] = sel_g

    # Filter data by gender for cascading
    df_filtered_by_gender = df_raw.copy()
    if sel_g and 'CompetitionGender' in df_filtered_by_gender:
        df_filtered_by_gender = df_filtered_by_gender[df_filtered_by_gender['CompetitionGender'].isin(sel_g)]

    # FILTER 2: Match Category (depends on Gender)
    cat_options = sorted(df_filtered_by_gender['MatchCategory'].dropna().unique().tolist()) if 'MatchCategory' in df_filtered_by_gender.columns else []
    # Ensure consistent ordering: Knockout first, then League
    cat_options_ordered = []
    if 'Knockout Matches' in cat_options:
        cat_options_ordered.append('Knockout Matches')
    if 'League Matches' in cat_options:
        cat_options_ordered.append('League Matches')

    sel_cat = st.sidebar.multiselect("Match Category", options=cat_options_ordered, default=st.session_state['f_cat'])
    st.session_state['f_cat'] = sel_cat

    # Filter data by gender + category for cascading
    df_filtered_by_cat = df_filtered_by_gender.copy()
    if sel_cat and 'MatchCategory' in df_filtered_by_cat:
        df_filtered_by_cat = df_filtered_by_cat[df_filtered_by_cat['MatchCategory'].isin(sel_cat)]

    # FILTER 3: Match Type (depends on Gender + Category)
    mt_options = sorted(df_filtered_by_cat['MatchType'].dropna().unique().tolist()) if 'MatchType' in df_filtered_by_cat.columns else []
    sel_mt = st.sidebar.multiselect("Match Type", options=mt_options, default=st.session_state['f_mt'])
    st.session_state['f_mt'] = sel_mt

    # Filter data by gender + category + match type for cascading
    df_filtered_by_mt = df_filtered_by_cat.copy()
    if sel_mt and 'MatchType' in df_filtered_by_mt:
        df_filtered_by_mt = df_filtered_by_mt[df_filtered_by_mt['MatchType'].isin(sel_mt)]

    # FILTER 4: Match Order (depends on Gender + Category + Match Type)
    mo_options = sorted(df_filtered_by_mt['MatchOrder'].dropna().unique().tolist()) if 'MatchOrder' in df_filtered_by_mt.columns else []
    sel_mo = st.sidebar.multiselect("Match Order", options=mo_options, default=st.session_state['f_mo'])
    st.session_state['f_mo'] = sel_mo

    # FILTER 5: Date Range (independent)
    dr = st.sidebar.date_input("Date Range", value=st.session_state['f_range'] if all(st.session_state['f_range']) else None, min_value=dmin, max_value=dmax)
    if isinstance(dr, tuple) and len(dr) == 2:
        st.session_state['f_range'] = dr

    # Apply all filters to get final filtered data
    with st.spinner("Applying filters..."):
        df = df_raw.copy()
        if sel_g and 'CompetitionGender' in df:
            df = df[df['CompetitionGender'].isin(sel_g)]
        if sel_cat and 'MatchCategory' in df:
            df = df[df['MatchCategory'].isin(sel_cat)]
        if sel_mt and 'MatchType' in df:
            df = df[df['MatchType'].isin(sel_mt)]
        if sel_mo and 'MatchOrder' in df:
            df = df[df['MatchOrder'].isin(sel_mo)]
        if all(st.session_state['f_range']) and 'MatchDate' in df:
            start, end = st.session_state['f_range']
            df = df[(df['MatchDate'] >= start) & (df['MatchDate'] <= end)]

        # Remove the temporary MatchCategory column from display
        if 'MatchCategory' in df.columns:
            df = df.drop(columns=['MatchCategory'])

    total_matches = len(df_raw); filtered_matches = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{filtered_matches}")
    c2.metric("Competitions", f"{df['CompetitionName'].nunique() if 'CompetitionName' in df else 0}")
    uniq_officials = 0
    for col in ['GroundUmpire1','GroundUmpire2','Referee']:
        if col in df: uniq_officials += df[col].dropna().astype(str).nunique()
    c3.metric("Officials (unique approx)", f"{uniq_officials}")

    st.subheader("📊 Match Details")
    st.write(f"Rows: {len(df)}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("👨‍⚖️ Umpire Workload Analysis")
    with st.spinner("Computing umpire pivot..."):
        ump_pivot = _pivot_umpires(df) if not df.empty else pd.DataFrame()
    st.dataframe(ump_pivot, use_container_width=True, hide_index=True)

    st.subheader("🏏 Referee Workload Analysis")
    with st.spinner("Computing referee pivot..."):
        ref_pivot = _pivot_referees(df) if not df.empty else pd.DataFrame()
    st.dataframe(ref_pivot, use_container_width=True, hide_index=True)

    st.subheader("📥 Download Filtered Data")
    if df.empty:
        st.caption("Nothing to download – adjust filters.")
    else:
        out = BytesIO()
        _write_formatted_excel(out, df, ump_pivot, ref_pivot)
        out.seek(0)
        st.download_button(
            label="Download Excel",
            data=out,
            file_name="bcci_filtered_scraped.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )



def run():
    # Single-mode: Real-time scraping app only
    realtime_scraper_app()

if __name__ == "__main__":
    run()


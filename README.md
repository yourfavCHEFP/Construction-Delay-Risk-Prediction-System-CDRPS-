# Construction Delay Risk Prediction System (CDRPS)

The Construction Delay Risk Prediction System (CDRPS) is a data-driven analytical platform for exploring construction delay risk, identifying patterns in project survey data, and visualizing risk signals through an interactive Streamlit dashboard.

## Key Features

- CSV upload with automatic encoding detection
- Data cleaning and numeric preprocessing
- PCA-based dimensionality reduction
- KMeans clustering for respondent grouping
- Delay Risk Index calculation
- Interactive dashboard with cluster and risk views

## Repository Layout

- `app.py` - Streamlit dashboard entry point
- `run_streamlit.bat` - Windows launcher for the dashboard
- `run_streamlit.ps1` - PowerShell launcher for the dashboard
- `CDRPS/` - project package and notebook workspace
- `requirements.txt` - Python dependencies

## How It Works

1. Upload a CSV file in the dashboard.
2. The app detects the file encoding and loads the data safely.
3. Numeric fields are cleaned, imputed, and scaled.
4. PCA reduces the feature space to two dimensions.
5. KMeans assigns cluster labels.
6. A Delay Risk Index is computed from the cleaned numeric features.
7. The dashboard presents the PCA view, cluster profile, risk distribution, and raw data table.

## Run the Dashboard

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

Launch Streamlit:

```powershell
streamlit run app.py
```

Or use the bundled launchers:

```powershell
.\run_streamlit.ps1
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, branching strategy, and commit guidance.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the planned delivery phases.

## License

See [LICENSE](LICENSE) for license details.

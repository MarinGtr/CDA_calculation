# 🚴 Velodrome CdA Analyzer

Professional aerodynamic drag coefficient analysis from velodrome testing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Features

- 📁 FIT file upload and parsing
- 🔍 Automatic interval detection (4× 1250m)
- 🔄 Turn detection with lean correction
- 📊 CdA estimation with uncertainty quantification
- 📈 Interactive visualizations
- 💾 Export results (CSV/JSON)

## Documentation

See `docs/` for detailed methodology and user guide.

## Testing

```bash
pytest tests/
```

## License

MIT License

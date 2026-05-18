# Chronic Kidney Disease (CKD) ML Project

A machine learning project for predicting and analyzing chronic kidney disease using various classification techniques.

## Project Structure

```
├── app/                    # Application code
├── datasets/              # Data files
│   └── kidney_disease.csv # Main dataset
├── images/                # Project images and visualizations
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks
│   └── CKD_Project.ipynb  # Main analysis notebook
├── reports/               # Generated reports and analysis
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CKD-ML-Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook notebooks/CKD_Project.ipynb
```

## Dataset

The project uses the Chronic Kidney Disease dataset from `datasets/kidney_disease.csv`.

## Technologies Used

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## License

This project is licensed under the MIT License - see the LICENSE file for details.

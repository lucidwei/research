
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Adjust font settings for Chinese characters
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # Specify default font to fix issue with Chinese characters not displaying
mpl.rcParams['axes.unicode_minus'] = False           # Fix issue with negative sign displaying as a square

# Load and process the data
data = pd.read_excel(r"D:\WPS_cloud\WPSDrive\13368898\WPS云盘\工作-麦高\杂活\北向_当日资金净流入.xlsx")
data_cleaned = data.iloc[5:].copy()
data_cleaned.columns = data.iloc[0]
data_cleaned.set_index("指标名称", inplace=True)
data_cleaned.index.name = "Date"
# data_cleaned.drop(columns=["指标名称"], inplace=True)
data_cleaned = data_cleaned.iloc[2:-2]
data_cleaned = data_cleaned.sort_index()
data_cleaned = data_cleaned.astype(float)
data_cleaned["万得全A涨跌幅"] = data_cleaned["万得全A"].pct_change()
data_cleaned = data_cleaned.fillna(0)
data_cleaned["10th_percentile"] = data_cleaned["万得全A"].rolling(window=252).quantile(0.10)
data_cleaned["90th_percentile"] = data_cleaned["万得全A"].rolling(window=252).quantile(0.90)
data_cleaned["Position"] = "Middle"
data_cleaned.loc[data_cleaned["万得全A"] <= data_cleaned["10th_percentile"], "Position"] = "Low"
data_cleaned.loc[data_cleaned["万得全A"] >= data_cleaned["90th_percentile"], "Position"] = "High"
data_cleaned = data_cleaned.sort_index()

import statsmodels.api as sm

# Initialize an empty dictionary to store results
results = {}
positions = ['Low', 'Middle', 'High']
# Iterate over the positions and calculate the linear regression model
for position in positions:
    subset = data_cleaned[data_cleaned["Position"] == position]

    # Define the independent variable (with a constant) and the dependent variable
    X = sm.add_constant(subset["北向:当日资金净流入"])
    y = subset["万得全A涨跌幅"]

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Extract relevant metrics
    correlation = subset["北向:当日资金净流入"].corr(subset["万得全A涨跌幅"])
    coef = model.params["北向:当日资金净流入"]
    conf_int = model.conf_int().loc["北向:当日资金净流入"].tolist()
    # Extract R-squared value
    r_squared = model.rsquared
    p_value = model.pvalues["北向:当日资金净流入"]

    results[position] = {
        "Correlation": correlation,
        "Coefficient": coef,
        # "Confidence Interval (95%)": conf_int,
        "R-squared": r_squared,
        "P-value": p_value
    }

results_df = pd.DataFrame(results).T
a=results_df*100
print(results_df)


# Plot the scatter plots for each position
fig, axes = plt.subplots(nrows=3, figsize=(10, 15), sharex=True)
# Determine common x-axis limits and ticks for all plots
global_xlim = (
    data_cleaned["北向:当日资金净流入"].min(),
    data_cleaned["北向:当日资金净流入"].max()
)
xticks = range(int(global_xlim[0]), int(global_xlim[1]) + 50, 50)
for ax, position in zip(axes, positions):
    subset = data_cleaned[data_cleaned["Position"] == position]
    sns.regplot(x="北向:当日资金净流入", y="万得全A涨跌幅", data=subset, ax=ax, scatter_kws={'s': 10, 'alpha': 0.3},
                line_kws={'color': 'red'})
    ax.set_title(f"Position: {position}")
    ax.set_ylabel("万得全A涨跌幅 (%)")

    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Set the x-axis limits, ticks, and label only for the bottom subplot
    ax.set_xlim(global_xlim)
    ax.set_xticks(xticks)
    if ax != axes[-1]:
        ax.set_xlabel('')
    else:
        ax.set_xlabel("北向:当日资金净流入 (亿元)")

    # Add dashed lines for x=0 and y=0
    ax.axvline(0, color='black', linestyle='--', lw=0.5)
    ax.axhline(0, color='black', linestyle='--', lw=0.5)

plt.tight_layout()
plt.show()

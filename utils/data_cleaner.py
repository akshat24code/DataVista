from ydata_profiling import ProfileReport

def create_eda_report(df, output_file="eda_report.html"):
    profile = ProfileReport(df, title="DataVista - EDA Report", explorative=True)
    profile.to_file(output_file)
    return output_file

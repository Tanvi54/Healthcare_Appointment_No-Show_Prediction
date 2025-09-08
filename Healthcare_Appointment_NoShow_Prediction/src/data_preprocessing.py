import pandas as pd

def preprocess_data(input_path="data/raw/Medical.csv", 
                    output_path="data/processed/noshow_clean_for_powerbi.csv"):
    # Load raw data
    df = pd.read_csv(input_path)
    df = df.drop_duplicates()
    # Convert datetime columns
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Clean target column: "No-show" (Yes/No → 1/0)
    df['No-show'] = df['No-show'].map({"No": 0, "Yes": 1})

    # Feature engineering
    df['WaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    # Drop irrelevant columns if needed
    df = df.drop(columns=['PatientId', 'AppointmentID'])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()



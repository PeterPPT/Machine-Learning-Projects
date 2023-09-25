

class Risk_score():
    def __init__(self):
        pass

    # Define the NEWS scoring criteria
    def calculate_news_score(self,row):
        score = 0

        # Heart Rate
        if row['HR'] <= 40:
            score += 3
        elif row['HR'] > 40 and row['HR'] <= 50:
            score += 1
        elif row['HR'] > 90 and row['HR'] <= 110:
            score += 1
        elif row['HR'] > 110 and row['HR'] <= 130:
            score +=2
        elif row['HR'] >= 131:
            score +=3

        # Respiratory Rate
        if row['Resp'] <= 8:
            score += 3
        elif row['Resp'] > 8 and row['Resp'] <= 11:
            score += 1
        elif row['Resp'] > 20 and row['Resp'] <= 24:
            score += 2
        elif row['Resp'] >= 25:
            score +=3

        # Systolic Blood Pressure
        if row['SBP'] <= 90:
            score += 3
        elif row['SBP'] > 90 and row['SBP'] <= 100:
            score += 2
        elif row['SBP'] >= 220:
            score +=3

        # Oxygen Saturation
        if row['O2Sat'] <= 91:
            score += 3
        elif row['O2Sat'] > 91 and row['O2Sat'] <= 93:
            score += 2
        elif row['O2Sat'] > 93 and row['O2Sat'] <= 95:
            score += 1

        # Temperature
        if row['Temp'] <= 35:
            score += 3
        elif row['Temp'] > 35 and row['Temp'] <= 36:
            score += 1
        elif row['Temp'] > 38 and row['Temp'] <= 39:
            score += 1
        elif row['Temp'] > 39:
            score +=2

        return score
    
    # Calculate SIRs score
    def calculate_sirs_criteria(self, row):
        score = 0  # score
    
        # Vital sign criteria
        if row['Temp'] > 38.0 or row['Temp'] < 36.0:
            score += 1
        if row['HR'] > 90:
            score += 1
        if row['Resp'] > 20 or row['PaCO2'] < 32:
            score += 1
        if row['WBC'] > 12 or row['WBC'] < 4:
            score += 1
        
        if score >= 2:
            criteria_met = 1.0
        else:
            criteria_met = 0.0

        return criteria_met

    # Calculate risk score
    def Calculate_risk_score(self, df):

        temp_df = df[['HR', 'O2Sat', 'SBP', 'Temp', 'Resp']].copy()
        # Calculate the NEWS score for each row
        df['NEWS_score'] = temp_df.apply(self.calculate_news_score, axis=1)

        # Calculate SIRS for each row
        temp_df = df[['Temp', 'HR', 'Resp', 'WBC', 'PaCO2']].copy()
        df['SIR_score'] = temp_df.apply(self.calculate_sirs_criteria, axis=1)

        return df











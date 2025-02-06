
def is_sweep_capactive(df,key):
    current_at_zero_voltage = df.loc[df['voltage'] == 0, 'current']
    max_current = df['current'].max()
    print("++")
    print(key)
    print("max_current" ,max_current)
    print("max_current" ,max_current /1000)
    print(current_at_zero_voltage.iloc[1])
    print("++")
    # print(current_at_zero_voltage)
    # print(current_at_zero_voltage[1])
    if current_at_zero_voltage.iloc[1] > max_current /1000:
        if current_at_zero_voltage.iloc[1] <= 1E-12:
            return True
        else:
            return False

    else:
        return False
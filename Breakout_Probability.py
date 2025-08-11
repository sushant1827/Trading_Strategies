
def calculate_breakout_probabilities(df, perc, num_levels=5, disable_zero_percent=False):
    """
    Calculate breakout probabilities for a given percentage step.
    
    Parameters:
        df (pd.DataFrame): Historical price data with columns ['open', 'high', 'low', 'close'].
        perc (float): Percentage step to calculate the levels.
        num_levels (int): Number of levels to calculate.
        disable_zero_percent (bool): If True, will not display 0% levels.

    Returns:
        dict: Breakout probabilities for the first percentage step levels in up and down direction.
    """
    # Calculate the percentage step
    step = perc / 100
    
    # Initialize counters
    green_count = 0
    red_count = 0
    green_high_hit = 0
    green_low_hit = 0
    red_high_hit = 0
    red_low_hit = 0

    # Process the data
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        # Determine if the candle is green or red
        green = curr_row['close'] > curr_row['open']
        red = curr_row['close'] < curr_row['open']
        
        if green:
            green_count += 1
            if curr_row['high'] >= prev_row['high'] + (step * prev_row['close']):
                green_high_hit += 1
            if curr_row['low'] <= prev_row['low'] - (step * prev_row['close']):
                green_low_hit += 1

        if red:
            red_count += 1
            if curr_row['high'] >= prev_row['high'] + (step * prev_row['close']):
                red_high_hit += 1
            if curr_row['low'] <= prev_row['low'] - (step * prev_row['close']):
                red_low_hit += 1

    # Calculate probabilities
    probabilities = {}
    if green_count > 0:
        probabilities['up_probability'] = (green_high_hit / green_count) * 100
        if not disable_zero_percent or (probabilities['up_probability'] > 0):
            probabilities['up_probability'] = probabilities.get('up_probability', 0.0)
        else:
            probabilities['up_probability'] = 0.0
    
    if red_count > 0:
        probabilities['down_probability'] = (red_low_hit / red_count) * 100
        if not disable_zero_percent or (probabilities['down_probability'] > 0):
            probabilities['down_probability'] = probabilities.get('down_probability', 0.0)
        else:
            probabilities['down_probability'] = 0.0

    return probabilities, green_count, green_high_hit, green_low_hit, red_count, red_high_hit, red_low_hit


def get_breakout_probability(open_price, high_price, low_price, close_price, prev_close, probabilities, 
                             green_count, green_high_hit, green_low_hit, 
                             red_count, red_high_hit, red_low_hit, step):
    """
    Get breakout probability for new values of open, high, low, and close.

    Parameters:
        open_price (float): The opening price.
        high_price (float): The highest price.
        low_price (float): The lowest price.
        close_price (float): The closing price.
        prev_close (float): The previous closing price.
        probabilities (dict): Pre-calculated probabilities from historical data.
        step (float): The percentage step used to calculate levels.

    Returns:
        dict: Breakout probabilities for the given open, high, low, close values.
    """
    # breakout_probs = {
    #     'up_probability': 0.0,
    #     'down_probability': 0.0
    # }
    
    # Determine if the candle is green or red
    green = close_price > open_price
    red = close_price < open_price
    
    if green:
        green_count += 1
        if high_price >= high_price + (step * prev_close):
            green_high_hit += 1
            # breakout_probs['up_probability'] = probabilities.get('up_probability', 0.0)
        if low_price <= low_price - (step * prev_close):
            green_low_hit += 1
            # breakout_probs['down_probability'] = probabilities.get('down_probability', 0.0)
            
    if red:
        red_count += 1
        if high_price >= high_price + (step * prev_close):
            red_high_hit += 1
            # breakout_probs['up_probability'] = probabilities.get('up_probability', 0.0)
        if low_price <= low_price - (step * prev_close):
            red_low_hit += 1
            # breakout_probs['down_probability'] = probabilities.get('down_probability', 0.0)

    # Calculate probabilities
    probabilities = {}
    if green_count > 0:
        probabilities['up_probability'] = (green_high_hit / green_count) * 100
        if (probabilities['up_probability'] > 0):
            probabilities['up_probability'] = probabilities.get('up_probability', 0.0)
        else:
            probabilities['up_probability'] = 0.0
    
    if red_count > 0:
        probabilities['down_probability'] = (red_low_hit / red_count) * 100
        if (probabilities['down_probability'] > 0):
            probabilities['down_probability'] = probabilities.get('down_probability', 0.0)
        else:
            probabilities['down_probability'] = 0.0
            
    return probabilities, green_count, green_high_hit, green_low_hit, red_count, red_high_hit, red_low_hit



# probabilities, green_count, green_high_hit, green_low_hit, red_count, red_high_hit, red_low_hit = calculate_breakout_probabilities(past_df, perc=0.01, num_levels=5, disable_zero_percent=False)
# print(probabilities)

# probabilities, green_count, green_high_hit, green_low_hit, red_count, red_high_hit, red_low_hit = get_breakout_probability(first_m_open, prev_m_high, prev_m_low, last_m_close, 
#                                          last_n_close, probabilities, green_count, green_high_hit, green_low_hit, 
#                                          red_count, red_high_hit, red_low_hit, step=0.0001)
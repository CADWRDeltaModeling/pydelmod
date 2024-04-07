from datetime import datetime, timedelta


def parse_military_date(date_str):
    # Replace "2400" with "0000" for compatibility with datetime parsing
    if date_str.endswith("2400"):
        date_str = date_str.replace("2400", "0000")
        # Parse the date
        date_obj = datetime.strptime(date_str, "%d%b%Y %H%M")
        # Add one day to the date since "2400" is the end of the specified day
        date_obj += timedelta(days=1)
    else:
        date_obj = datetime.strptime(date_str, "%d%b%Y %H%M")

    return date_obj

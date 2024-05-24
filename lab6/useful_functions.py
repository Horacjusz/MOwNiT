def time_sentence(total_time, header="", precision = 3):
    # Adjust precision to a multiple of 3
    val = precision % 3
    if val != 0 :
        precision += 3 - val
    
    eps = 10**-precision
    
    total_time = round(total_time,precision)
    
    def create_sentence(value, text):
        if value == 0:
            return ""
        if value == 1:
            return f" {value} {text}"
        return f" {value} {text}s"
    
    names = {}
    names[3] = "millisecond"
    names[6] = "microsecond"
    names[9] = "nanosecond"
    names[12] = "picosecond"
    names[15] = "femtosecond"
    names[18] = "attosecond"
    names[21] = "zeptosecond"
    names[24] = "yoctosecond"
    names[27] = "rontosecond"
    names[30] = "quectosecond"
    
    dividers = [(31536000, "year"), (86400, "day"), (3600, "hour"), (60, "minute"), (1, "second"), (10**-3, names[3]), (10**-6, names[6]), (10**-9, names[9]), (10**-12, names[12]), (10**-15, names[15]), (10**-18, names[18]), (10**-21, names[21]), (10**-24, names[24]), (10**-27, names[27]), (10**-30, names[30])]
    
    time_str = ""
    
    for divider,text in dividers :
        current_time = min(999,int(total_time // divider))
        total_time = total_time - current_time * divider
        time_str += create_sentence(int(current_time), text)
        if total_time <= eps : break

    if time_str == "" : time_str = " less than 1 " + names[precision]

    return header + time_str
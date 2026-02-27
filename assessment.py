def assess_damage(damage_list):
    severity_score = 0

    for damage in damage_list:
        if damage == "dent":
            severity_score += 2
        elif damage == "scratch":
            severity_score += 1
        elif damage == "crack":
            severity_score += 3
        elif damage == "glass shatter":
            severity_score += 5
        elif damage == "lamp broken":
            severity_score += 4
        elif damage == "tire flat":
            severity_score += 5

    if severity_score <= 3:
        level = "Low"
    elif severity_score <= 7:
        level = "Medium"
    else:
        level = "High"

    return severity_score, level

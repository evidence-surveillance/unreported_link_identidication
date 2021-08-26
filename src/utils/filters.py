def get_suitable_completion_date(time_data):
    ans = []
    for i, date in enumerate(time_data):
        if date:
            if len(date) >= 6:
                month = date[:3]
                year = date[-4:]
                if year.isdigit() and \
                        (int(year) > 2007 or (year == '2007' and month in ['Oct', 'oct', 'Nov', 'nov', 'Dec', 'dec'])):
                    ans.append(i)
                else:
                    print(date)
            elif len(date) >= 4:
                year = date[-4:]
                if year.isdigit() and int(year) > 2007:
                    ans.append(i)
                else:
                    print(date)
            else:
                print(date)
    return ans


def get_publish_after_complete(ctime, ptime, fid=None):
    assert len(ctime) == len(ptime)
    ans = []
    for i, (ct, pt) in enumerate(zip(ctime, ptime)):
        if ct and pt:
            if len(ct) >= 4:
                year = ct[-4:]
                if year.isdigit() and 2007 < int(year) <= pt:
                    ans.append(i)
                else:
                    print(ct, pt)
            else:
                print(ct, pt)
    if fid:
        return sorted(list(set(ans) & set(fid)))
    else:
        return ans


def filter_db(db, col_name, val):
    return [i for i, s in enumerate(db[col_name]) if s == val]


def get_complete_trial_id(db):
    return filter_db(db, 'recruitment_status', 'Completed')
#     return [i for i, s in enumerate(db['recruitment_status']) if s=='Completed']


def get_interventional_trial_id(db):
    return filter_db(db, 'study_type', 'Interventional')
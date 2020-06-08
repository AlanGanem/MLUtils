import itertools
import pandas as pd
import numpy as np
import tqdm
import json
# IO part
def update_rules_json(new_rule, RULES_JSON_PATH):
    RULES_DICT = import_rules_dict(RULES_JSON_PATH)
    if list(new_rule)[0] in list(RULES_DICT):
        raise ValueError('Rule with ID {} already exists'.format(list(new_rule)[0]))
    else:
        RULES_DICT = {**RULES_DICT, **new_rule}

    # saves new rules
    with open(RULES_JSON_PATH, 'w') as outfile:
        json.dump(RULES_DICT, outfile)
    print('Old rules_json file has been overwritten.')
    return
##################################
#                                #
#                                #
##################################
def import_rules_dict(RULES_JSON_PATH):
    # open rules json
    with open(RULES_JSON_PATH) as json_file:
        rules_dict = dict(json.load(json_file))
    return rules_dict


def crete_rule(rule_json, user, date):
    rule_id = str(user) + '_' + str(date)
    rule = {rule_id: rule_json}
    return rule

def check_rule(rules_dict):

    #
    all_rules = {}
    for id_ in rules_dict:
        features = [rules_dict[id_]['condition'][feature] for feature in rules_dict[id_]['condition']]
        all_rules[id_] = set(itertools.product(*features))

    rules_list = [all_rules[id_] for id_ in all_rules]
    intersections = set.intersection(*rules_list)
    intersections_by_id = {id_: (all_rules[id_] & intersections) for id_ in all_rules}
    id_by_intersection = {comb: [] for comb in intersections}
    for id_ in intersections_by_id:
        for comb in intersections_by_id[id_]:
            id_by_intersection[comb].append(id_)

    return rules_dict


def create_rule_masks(df, rules_json):
    '''
    Apply rules to dependent var in df (only & operator suported)

    :param df: dataframe to apply rules
    :param rules_json: json or dict containing rules (each rule is also a dictionary containinf values of features)
    :return: df with new values of IVAMIRO
    '''

    # create msk_dict

    msk_dict = {}
    for id_ in tqdm.tqdm(rules_json):

        i = 0
        for feature in rules_json[id_]['condition']:

            if i == 0:
                msk = df[feature].isin(rules_json[id_]['condition'][feature])
            else:
                msk = msk & df[feature].isin(rules_json[id_]['condition'][feature])
            i += 1

        msk_dict[id_] = {'mask': msk, 'value': rules_json[id_][
            'implication']}  # rules_json[id_]['implication'] is a dict of {<feature>:[<value>]}

    return msk_dict


def apply_rules(df, rules_json):
    '''
    Apply rules to dataframe
    :param df:
    :param mask_dict:
    :return:
    '''

    mask_dict = create_rule_masks(df, rules_json)
    df = df.copy()
    df['model_rule_id'] = np.nan
    for id_ in tqdm.tqdm(mask_dict):
        for feature in mask_dict[id_]['value']:
            df.loc[mask_dict[id_]['mask'], feature] = mask_dict[id_]['value'][feature]
            df.loc[mask_dict[id_]['mask'], 'model_rule_id'] = id_

    return df


def show_rules_df(rules_json):
    #rules_dict = import_rules_dict(RULES_JSON_PATH)
    df = pd.concat({k: pd.DataFrame(v) for k, v in (check_rule(rules_json=rules_json)).items()})
    df.index.set_names(['id', 'column'], inplace=True)
    return df
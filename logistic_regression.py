import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    # preparation
    train = pd.read_csv('train.txt', sep=' ', header=None, names=['token', 'POS', 'chunking', 'chunking2'])
    train = train[['token','POS']]
    train = add_features(train)
    trainx, trainy, testx, testy = format(train)
    # trainx = train.drop(['token','prev_token','next_token','POS','prev_POS','next_POS'], axis=1).to_numpy()
    # trainy = train['POS'].to_numpy()
    
    # # training
    print('Training...')
    model = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=100).fit(trainx, trainy)
    print('Finished Training.')
    
    # evaluation
    pred = model.predict(testx)
    acc = get_acc(pred, testy)
    print('Accuracy:', round(acc*100,2), '%')
    
    # acc_by_class = get_acc_by_class(pred, testy)
    # print(acc_by_class)

    # predict tokens for unlabeled test file
    with open('unlabeled_test_test.txt', 'r') as file:
        test_tokens = file.readlines()
        test_tokens = [line[:-1] for line in test_tokens]
    test = pd.DataFrame({'token': test_tokens})
    test = add_features(test)
    test = test.drop(['token','prev_token','next_token'], axis=1).to_numpy()
    output = model.predict(test)

    create_output_file(test_tokens, output)



def create_output_file(test_tokens, output):
    with open('GPTeam.test.txt', 'w') as f:
        for i, token in enumerate(test_tokens):
            if token == "":
                f.write('\n')
            else:
                f.write(f'{token} {output[i]}\n')



def add_features(df):    
    # add non-context-dependent features
    df['capitalized'] =           df['token'].apply(lambda x: 1 if str(x).istitle() else 0)
    df['word_length'] =           df['token'].apply(lambda x: len(str(x)))
    df['and_or_but'] =            df['token'].apply(lambda x: 1 if str(x) in ['and','or','but'] else 0)
    df['a_an_the'] =              df['token'].apply(lambda x: 1 if str(x) in ['a','and','the'] else 0)
    df['to'] =                    df['token'].apply(lambda x: 1 if str(x) == 'to' else 0)
    df['colon'] =                 df['token'].apply(lambda x: 1 if str(x) in [':',';','--'] else 0)
    df['comma'] =                 df['token'].apply(lambda x: 1 if str(x) == ',' else 0)
    df['period'] =                df['token'].apply(lambda x: 1 if str(x) == '.' else 0)
    df['dollar_sign'] =           df['token'].apply(lambda x: 1 if str(x) == '$' else 0)
    df['quotes_reg'] =            df['token'].apply(lambda x: 1 if str(x) == '\'\'' else 0)
    df['quotes_slant'] =          df['token'].apply(lambda x: 1 if str(x) == '``' else 0)
    df['contains_number'] =       df['token'].apply(lambda x: 1 if any(chr.isdigit() for chr in str(x)) else 0)
    df['contains_apostrophe'] =   df['token'].apply(lambda x: 1 if any(chr == '\'' for chr in str(x)) else 0)
    df['contains_hyphen'] =       df['token'].apply(lambda x: 1 if any(chr == '-' for chr in str(x)) else 0)
    df['equals'] =                df['token'].apply(lambda x: 1 if str(x) == '=' else 0)
    df['contains_pound'] =        df['token'].apply(lambda x: 1 if any(chr == '#' for chr in str(x)) else 0)
    df['have_had'] =              df['token'].apply(lambda x: 1 if str(x) in ['has','have','had'] else 0)
    df['be'] =                    df['token'].apply(lambda x: 1 if str(x) in ['is','are','was','were'] else 0)
    df['more_less'] =             df['token'].apply(lambda x: 1 if str(x) in ['more','less'] else 0)
    df['most_least'] =            df['token'].apply(lambda x: 1 if str(x) in ['most','least'] else 0)
    df['he_she_they'] =           df['token'].apply(lambda x: 1 if str(x) in ['he','she','they'] else 0)
    df['I_you_we'] =              df['token'].apply(lambda x: 1 if str(x) in ['I','you','we'] else 0)
    df['pre_dis_re_mis'] =        df['token'].apply(lambda x: 1 if str(x)[:3] in ["mis","dis"] or str(x)[:2] == "re" else 0)
    df['pre_un_in_il_im'] =       df['token'].apply(lambda x: 1 if str(x)[:2] in ['un','in','il','im'] else 0)
    df['pre_wh'] =                df['token'].apply(lambda x: 1 if str(x)[:2] == 'wh' else 0)
    df['suf_ed'] =                df['token'].apply(lambda x: 1 if str(x)[-2:] == "ed" else 0)
    df['suf_tion_sion_ment'] =    df['token'].apply(lambda x: 1 if str(x)[-4:] in ["tion", "sion", "ment"] else 0)
    df['suf_s'] =                 df['token'].apply(lambda x: 1 if str(x)[-1:] == "s" else 0)
    df['suf_or'] =                df['token'].apply(lambda x: 1 if str(x)[-2:] == "or" else 0)
    df['suf_er'] =                df['token'].apply(lambda x: 1 if str(x)[-2:] == "er" else 0)
    df['suf_est'] =               df['token'].apply(lambda x: 1 if str(x)[-3:] == "est" else 0)
    df['suf_able_ible_ful'] =     df['token'].apply(lambda x: 1 if str(x)[-4:] in ["able", "ible"] or str(x)[-3:] == 'ful' else 0)
    df['suf_ing'] =               df['token'].apply(lambda x: 1 if str(x)[-3:] == "ing" else 0)
    df['suf_ly'] =                df['token'].apply(lambda x: 1 if str(x)[-2:] == "ly" else 0)
    df['suf_y'] =                 df['token'].apply(lambda x: 1 if str(x)[-1:] == "y" else 0)
    df['suf_fy_ate_en_ize'] =     df['token'].apply(lambda x: 1 if str(x)[-3:] in ['ate','ize'] or str(x)[-2:] in ['fy','en'] else 0)
    df['suf_self'] =              df['token'].apply(lambda x: 1 if str(x)[-4:] == 'self' or str(x)[-6:] == 'selves' else 0)

    # add context-dependent variables
    empty_row = [0]*len(df.columns)
    # print(empty_row)
    
    prev_df = df.copy()
    prev_df.columns = ['prev_' + col for col in prev_df.columns]
    prev_df.loc[-1] = empty_row
    prev_df.index = prev_df.index + 1
    prev_df = prev_df.sort_index()
    prev_df = prev_df.drop(prev_df.index[-1])
    # print(prev_df)

    next_df = df.copy()
    next_df.columns = ['next_' + col for col in next_df.columns]
    next_df.loc[len(next_df)] = empty_row
    next_df = next_df.drop(next_df.index[0])
    next_df.index = next_df.index - 1
    # print(next_df)
    
    # concatenate dfs for previous, current, and next token
    return pd.concat([df, prev_df, next_df], axis=1)



def format(data):
    x = data.drop(['token','prev_token','next_token','POS','prev_POS','next_POS'], axis=1).to_numpy()
    y = data['POS'].to_numpy()
    i = int(.8*len(y))
    trainx, trainy, testx, testy = x[:i], y[:i], x[i:], y[i:]
    return trainx, trainy, testx, testy



def get_acc(pred, testy):
    correct=0
    for i, label in enumerate(testy):
        if pred[i]==label:
            correct+=1
    return correct/len(testy)

def get_acc_by_class(pred, testy):
    acc_by_class=dict()
    for i, label in enumerate(testy):
        if label not in acc_by_class:
            acc_by_class[label]=[0,0]
        if pred[i]==label:
            acc_by_class[label][0]+=1
        acc_by_class[label][1]+=1
    for label in acc_by_class:
        acc_by_class[label][0] = round(acc_by_class[label][0] / acc_by_class[label][1], 2)
    return dict(sorted(acc_by_class.items(), key=lambda item: item[1][1], reverse=True))



if __name__ == '__main__':
    main()

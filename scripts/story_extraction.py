
import pandas as pd
import argparse
import os

from utils import compare_and_merge_weeks, check_clusters_and_save

pd.options.mode.chained_assignment = None  # default='warn'

def week_loader(week, year):
    counter = "%02d" % (int(week))
    value = year + '-' + str(counter)
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2018', help='Year to bridge')
    parser.add_argument('--similarity_threshold', type=int, default=0.35, help='Threshold of similarity between articles')
    parser.add_argument('--merging_threshold', type=int, default=0.33, help='Threshold of proportion of articles between clusters')
    parser.add_argument('--min_articles', type=int, default=10, help='Minimum articles per week to keep looking')
    parser.add_argument('--start_week', type=int, default=-1, help='Week to start from')
    args = parser.parse_args()

    week_start = args.start_week-1
    week_look = 0
    first_time = True
    direction = 'forward'
    done_here = False
    finalized_df = pd.DataFrame(index=[1], columns=['final_labels'])
    finalized_df = finalized_df.fillna(0)  # with 0s rather than NaNs

    while not done_here:
        if first_time and direction == 'forward':
            print('option 1')
            first_time = False
            week_start += 1
            starting = week_loader(week_start, args.year)
            week_look = week_start + 1
            following = week_loader(week_look, args.year)
            if os.path.exists("Final/Chosen_clusters" + starting + ".csv"):
                origin = pd.read_csv("Final/Chosen_clusters" + starting + ".csv")
            else:
                print('No clean data for week {0}'.format(starting))
                finalized_df = check_clusters_and_save(finalized_df, forward, args.merging_threshold)
                done_here=True
                break
        elif not first_time and direction == 'forward':
            print('option 2')
            origin = forward
            week_look += 1
            following = week_loader(week_look, args.year)
        elif first_time and direction == 'backward' and week_look > 0:
            print('option 3')
            first_time = False
            origin = forward
            week_look = week_start-1
            following = week_loader(week_look, args.year)
        elif not first_time and direction == 'backward' and week_look > 0:
            print('option 4')
            origin = forward
            week_look -= 1
            following = week_loader(week_look, args.year)
        else:
            print('option 5')
            first_time = True
            direction = 'forward'
            print('Finished week {0}. Going to next week!'.format(week_start))
            finalized_df = check_clusters_and_save(finalized_df, forward, args.merging_threshold)
            continue
        if os.path.exists("Dirty/Clusters" + following + ".csv"):
            compare_to = pd.read_csv("Dirty/Clusters" + following + ".csv")
        else:
            print('No dirty data for week {0}'.format(following))
            if direction == 'forward':
                direction = 'backward'
                first_time = True
            else:
                direction = 'forward'
                first_time = True
                print('Finished week {0}. Going to next week!'.format(week_start))
                finalized_df = check_clusters_and_save(finalized_df, forward, args.merging_threshold)
            continue

        print('Comparing week {0} and week {1}'.format(week_start, week_look))

        forward, new = compare_and_merge_weeks(origin, compare_to, args.similarity_threshold, args.min_articles, finalized_df)

        if new.shape[0] == 0:
            if direction == 'forward':
                direction = 'backward'
                first_time = True
            else:
                direction = 'forward'
                first_time = True
                print('Finished week {0}. Going to next week!'.format(week_start))
                finalized_df = check_clusters_and_save(finalized_df, forward, args.merging_threshold)

    print(finalized_df.groupby(['final_labels'])['DIRECCION'].size())
    print(finalized_df.head())

    finalized_df = finalized_df.drop_duplicates(subset=["DIRECCION", "final_labels"])

    finalized_df.to_csv('b-final_links3.csv', index=False)
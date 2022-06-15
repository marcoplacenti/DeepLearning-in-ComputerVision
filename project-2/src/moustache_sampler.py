import random
import shutil

mustaches = []
not_mustaches = []
with open('./data/Anno/list_attr_celeba.txt') as infile:
    header = True
    data = []
    for line in infile:
        if header:
            keys = line.strip().split(" ")
            mustache_idx = keys.index("Mustache")
            male_idx = keys.index("Male")
            no_beard_idx = keys.index("No_Beard")
            header = False
        else:
            filename = line.strip().split(" ")[0]
            line_values = [value for value in line.strip().split(" ")[1:] if value != "" ]
            if line_values[mustache_idx] == "1" and line_values[male_idx] == "1" and line_values[no_beard_idx] == "-1":
                mustaches.append(filename)
            elif line_values[mustache_idx] == "-1" and line_values[male_idx] == "1"  and line_values[no_beard_idx] == "1":
                not_mustaches.append(filename)

samples_per_class = 25

mus_samples = random.sample(mustaches, samples_per_class)
not_mus_samples = random.sample(not_mustaches, samples_per_class)

for sample in mus_samples:
    shutil.copyfile('./data/img_align_celeba/'+sample, './data/samples/mustache/'+sample)

for sample in not_mus_samples:
    shutil.copyfile('./data/img_align_celeba/'+sample, './data/samples/no_mustache/'+sample)



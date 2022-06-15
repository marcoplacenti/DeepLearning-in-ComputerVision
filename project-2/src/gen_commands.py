
import os

cmd = []
for file in os.listdir('./data/samples/mustache/'):
    file_with_ext = file
    filename = file_with_ext.split(".")[0]

    cmd.append(f"python3 projector.py --outdir=samples/projections/mustache/{filename} --target=samples/mustache/{file_with_ext} --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --save-video=false")

for file in os.listdir('./data/samples/no_mustache/'):
    file_with_ext = file
    filename = file_with_ext.split(".")[0]

    cmd.append(f"python3 projector.py --outdir=samples/projections/no_mustache/{filename} --target=samples/no_mustache/{file_with_ext} --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --save-video=false")

with open("./data/cmds/commands.txt", "w") as outfile:
    for c in cmd:
        outfile.write(c+"\n")
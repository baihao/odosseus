You can run commands.
Put all files in the project folder /scratch/local/velikanov/work/odosseus.
You cannot run on the GPU directly. You need to connect to tmux: tmux -S /tmp/tmux-socket-12345 attach -t mysession .  Use this command "env CUDA_VISIBLE_DEVICES=0 bash" inside tmux to be able to access the gpu.

Use python only through miniconda env. You can create new envs.
Run commands through tmux only.
tmux -S /tmp/tmux-socket-12345 attach -t mysession
Make sure your scripts never consume more than 20 GB CPU RAM.
Make sure you don't run out of disk space.
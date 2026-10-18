You can run commands.
Put all files in the project folder /scratch/local/velikanov/work/odosseus.
You cannot run on the GPU directly. You need to connect to tmux: tmux -S /tmp/tmux-socket-12345 attach -t mysession .  Use this command "env CUDA_VISIBLE_DEVICES=0 bash" inside tmux to be able to access the gpu.

Use python only through miniconda env. You can create new envs.
Run commands through tmux only.
tmux -S /tmp/tmux-socket-12345 attach -t mysession
Make sure your scripts never consume more than 20 GB CPU RAM.
Make sure you don't run out of disk space.

## Loop pipeline
See the project description and data in data/ folder.

Run in a loop:
1. Select a method. Do websearch to find accurate information. Install it.
2. Run on the data, save output
3. Run visualization - save images, htmls, videos, whatever you find suitable. Choose the best visualization method for debugging
4. Visualize difference between GT track and the prediction. Calculate error, save a PDF report with metric values. If the error is not low enough, repeat the loop.

Document your progress in docs/ folder. Create one summary log, with the major changes,
in the format "[day-hour-minute] one line of description". For more detailed description, create readmes in outputs/ folder. Save experiment outputs in outputs/ folder, each experiment in its own folder. Save PDFs with metric values, save visualizations.

## Data location
New daytime frames located in:
data/rosbag_2026-10-13_16-08-46_cameras_lap

Look at data in 23-00-38_cut_120m_update
apt update
apt install -y tmux
git pull origin main
pip install -r requirements.txt
tmux new-session -d -s slave1
# tmux attach -t slave1
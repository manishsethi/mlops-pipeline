bash -c '
proj="mlops-pipeline" &&
mkdir -p $proj/{data/{raw,processed},models,src,tests,docker,monitoring,.github/workflows,.dvc} &&
touch \
  $proj/.github/workflows/ci-cd.yml \
  $proj/src/{train.py,predict.py,api.py} \
  $proj/docker/Dockerfile \
  $proj/requirements.txt \
  $proj/.gitignore \
  $proj/dvc.yaml \
  $proj/README.md &&
echo "# mlops-pipeline" > $proj/README.md &&
echo "*.pyc\n__pycache__/\n.env\nmlruns/\n" > $proj/.gitignore &&
echo -e "dvc\nmlflow\nflask\nscikit-learn\n" > $proj/requirements.txt
'
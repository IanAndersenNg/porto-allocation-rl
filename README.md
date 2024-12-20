# porto-allocation-rl

Our project is to optimize portofolio allocation using reinforcement learning methods. We aim to find the best combination of allocation between 2 different asset classes X and Y as a risk management solution. We consider both returns and risk as our performance criteria of the agent. Through this agent, we aim to capture insights and policies from historical data of these 2 asset classes.

## Execution Related

Two reward functions are provided: portfolio returns and differential sharpe ratio. To use portfolio returns as the reward function, simply run the model as normal. To use the differential sharpe ratio, add the dsr flag, e.g.:

```
python models/gradientTD.py --dsr
```

Run all the models with and without differential sharpe ratio. The result will be saved at the `result/` folder (have to create it manually if not exists):

```
python run.py [--dsr]
```

## Git Related

To make modifications to this project, please follow the steps below.

1. Use git pull to pull the repository down to your local computer, updating any changes from the remote repository. To do so, execute the following command:
   ```
   git pull
   ```
2. To add a new feature or change, checkout to a new branch before beginning your work. This practice will ensure that changes from main are always the source of truth, and reduce merge conflicts.

   ```
   git checkout -b branch_name_here
   ```

   - The code above creates a branch and switches to it. If the branch already exists, remove the -b flag from the command. For instance, if you want to checkout to main (but not create it, since it has already been created):

   ```
   git checkout main
   ```

3. Make your changes accordingly. Once that has been done, execute the following commands sequentially:

   ```
   git add .
   ```

   - This command adds all current files in your local system's project folder to be tracked by git, meaning that other git commands such as pushing will include these files. This includes any changes to existing files, as well as new files.

   ```
   git commit -m "add your commit message here"
   ```

   - This command creates a new commit, which is essentially a project modification history. The commit message is there to help other developers know the goals this specific commit aims to achieve.

   ```
   git push origin branch_name_here
   ```

   - This command pushes the changes in your commit to the branch called branch_name_here. As a best practice, please make sure that the name of the branch you created in your local system matches that of this command.
   - If the branch does not yet exist in git, git will automatically create a new branch. Otherwise, it will try to push your changes to the given branch.

4. Once your feature / changes have been complete and are ready to be merged to the main / master branch, navigate to github and add a pull request to merge your branch to main. Then, select the individuals to review the code, and once it has been reviewed and is ready to merge, submit the pull request.

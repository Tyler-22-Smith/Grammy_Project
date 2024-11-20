# Grammy_Project

## Running Data Scraping Script "SpotifyData.qmd"
Requirements:
  - Spotify Developer Credentials
    - Login to spotify *in web browser* allow developer mode in settings
    - Find your spotify "client id" and "client secret id"
  - Create a local .txt file named "KEYS.txt" and input credentials
    - TEXT FILE NEEDS TO BE IN SAME LOCAL DIRECTORY AS .Rproj
    - SEE KEYS.txt SETTUP BELOW
    - .gitignore should prevent user credentials from being pushed to repo if changes are made
    - if KEYS.txt is pushing open gitbash and run command "git rm --cache KEYS.txt"

KEYS.txt Settup as followed (use your credentials):
variable: value
client_id: xxxxxxx
client_secret: xxxxxxx

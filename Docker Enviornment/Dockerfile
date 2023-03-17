FROM ubuntu:focal

#getting Enviornment Data
ENV GH_TOKEN="Git_Hub_Token"
ENV GH_EMAIL="Git_Hub_Email"
ENV GH_UNAME="Git_Hub_UserNamen"
ENV GH_REPO_OWNER = "Git_Hub_Repository_Owner"
ENV GH_REPO_NAME = "Git_Hub_Repository_Name"

RUN echo "GH_TOKEN:         $GH_TOKEN"
RUN echo "GH_EMAIL:         $GH_EMAIL"
RUN echo "GH_UNAME:         $GH_UNAME"
RUN echo "GH_REPO_OWNER:    $GH_REPO_OWNER"
RUN echo "GH_REPO_NAME:     $GH_REPO_NAME"

# apt update
RUN apt update


# install zsh
RUN apt install zsh -y
# install ohmyzsh and set to default shell
RUN yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"


# install git
RUN apt update -y && apt install git -y
# install gpg
RUN apt update -y && apt install gpg -y
# install curl
RUN type -p curl >/dev/null || apt install curl -y


# install GitHub CLI (gh)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y
# clone repo 
RUN gh repo clone ${GH_REPO_OWNER}/${GH_REPO_NAME}

# configure git to use GitHub CLI as a credential helper
RUN gh auth setup-git

# setup git
WORKDIR /java-masterclass
RUN git config gpg.program gpg
# RUN git config commit.gpgsign true
RUN git config user.name "${GH_UNAME}"
RUN git config user.email "${GH_EMAIL}"

CMD ["bash"]
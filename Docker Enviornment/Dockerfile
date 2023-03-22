FROM ubuntu:focal



ARG gh_token
ARG gh_email
ARG gh_uname
ARG gh_repo_owner
ARG gh_repo_name

ENV GH_TOKEN ${gh_token}

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
RUN gh repo clone ${gh_repo_owner}/${gh_repo_name}

# configure git to use GitHub CLI as a credential helper
RUN gh auth setup-git

# setup git
WORKDIR /${gh_repo_name}
RUN git config gpg.program gpg
# RUN git config commit.gpgsign true
RUN git config user.name "${gh_uname}"
RUN git config user.email "${gh_email}"

#install Dependecies
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# installing python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.5 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio autoPyTorch


CMD ["bash"]
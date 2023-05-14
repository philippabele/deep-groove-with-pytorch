FROM ubuntu:latest

ARG gh_token
ARG gh_email
ARG gh_uname
ARG gh_repo_owner
ARG gh_repo_name
ARG gh_branch

ENV GH_TOKEN ${gh_token}

# install zsh
RUN apt-get update
RUN apt install zsh -y
RUN yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# install git, gpg and curl
RUN apt update -y && apt install git -y
RUN apt update -y && apt install gpg -y
RUN type -p curl >/dev/null || apt install curl -y

# install GitHub CLI (gh)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y


# config GitHub  Repository and cloning the Project

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

RUN git checkout ${gh_branch}

# install linux packages
RUN apt-get update

# Setup the autoPyTorch Enviornment
# workaround for https://github.com/automl/auto-sklearn/issues/867
RUN apt-get -y install locales
RUN touch /usr/share/locale/locale.alias
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG _US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt install -y build-essential

RUN apt install -y swig

RUN pip3 install autoPyTorch

RUN apt install libgl1-mesa-glx -y
RUN apt-get install libglib2.0.0 -y

CMD ["bash"]
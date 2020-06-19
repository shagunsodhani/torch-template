# type: ignore
import nox


@nox.session(venv_backend="conda")
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements.txt")
    session.run("flake8", "codes")
    session.run("flake8", "main.py")
    session.run("black", "--check", "codes")
    session.run("black", "--check", "main.py")
    session.run("yamllint", "config")


@nox.session(venv_backend="conda")
def mypy(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements.txt")
    session.run("mypy", "--strict", "codes")
    session.run("mypy", "--strict", "main.py")

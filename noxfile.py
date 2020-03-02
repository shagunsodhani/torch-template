# type: ignore
import nox


@nox.session(venv_backend="conda")
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements.txt")
    session.run("flake8", "codes")
    session.run("black", "--check", "codes")
    session.run("mypy", "--strict", "codes")

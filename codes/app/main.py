"""This is the main entry point for the code"""

from codes.grad_student.checkpointable_grad_student import CheckpointableGradStudent
from codes.utils.argument_parser import argument_parser
from codes.utils.util import timing


@timing
def run(config_id):
    """Run the code"""

    grad_student = CheckpointableGradStudent(config_id)
    grad_student.run()


if __name__ == "__main__":
    run(
        config_id=argument_parser()
    )

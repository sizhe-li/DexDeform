import argparse
import sys

from tracking.LeapSDK import Leap
from tracking.lib import *

HEADLESS = False


def main(args):
    # Create a listener

    if args.dual:
        if args.float:
            listener = DualShadowHandFloatBase(headless=HEADLESS)
        else:
            listener = DualShadowHandFixedBase(headless=HEADLESS)
    else:
        if args.float:
            listener = ShadowHandFloatBase(headless=HEADLESS, isLeftHand=args.left)
        else:
            listener = ShadowHandFixedBase(headless=HEADLESS, isLeftHand=args.left, with_object=args.object)

    # Create a controller
    controller = Leap.Controller()

    # Connect listener and controller
    controller.add_listener(listener)

    # Simple loop
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove listener
        controller.remove_listener(listener)


if __name__ == "__main__":
    # Check for sample / test mode
    parser = argparse.ArgumentParser(
        description="Run LeapMotionListener with/without sampling/testing mode"
    )
    parser.add_argument("-f",
                        "--float",
                        help="floating hand",
                        action="store_true",
                        required=False)
    parser.add_argument("-l",
                        "--left",
                        help="using left hand",
                        action="store_true",
                        required=False)
    parser.add_argument("-d",
                        "--dual",
                        help="using dual hands",
                        action="store_true",
                        required=False)
    parser.add_argument("-o",
                        "--object",
                        help="with object",
                        action="store_true",
                        required=False)


    args = parser.parse_args()

    main(args)

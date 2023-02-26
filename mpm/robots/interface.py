from typing import Dict, List

import numpy as np

FINGERTIP_SITE_NAMES = [
    "robot0:S_fftip",
    "robot0:S_mftip",
    "robot0:S_rftip",
    "robot0:S_lftip",
    "robot0:S_thtip",
]

JOINTS = [
    "robot0:WRJ1",  # joint_id 00, actuator_id 00
    "robot0:WRJ0",  # joint_id 01, actuator_id 01
    "robot0:FFJ3",  # joint_id 02, actuator_id 02
    "robot0:FFJ2",  # joint_id 03, actuator_id 03
    "robot0:FFJ1",  # joint_id 04, actuator_id 04, tendon "FFT1", coupled joint
    "robot0:FFJ0",  # joint_id 05, actuator_id 04, tendon "FFT1", coupled joint
    "robot0:MFJ3",  # joint_id 06, actuator_id 05
    "robot0:MFJ2",  # joint_id 07, actuator_id 06
    "robot0:MFJ1",  # joint_id 08, actuator_id 07, tendon "MFT1", coupled joint
    "robot0:MFJ0",  # joint_id 09, actuator_id 07, tendon "MFT1", coupled joint
    "robot0:RFJ3",  # joint_id 10, actuator_id 08
    "robot0:RFJ2",  # joint_id 11, actuator_id 09
    "robot0:RFJ1",  # joint_id 12, actuator_id 10, tendon "RFT1", coupled joint
    "robot0:RFJ0",  # joint_id 13, actuator_id 10, tendon "RFT1", coupled joint
    "robot0:LFJ4",  # joint_id 14, actuator_id 11
    "robot0:LFJ3",  # joint_id 15, actuator_id 12
    "robot0:LFJ2",  # joint_id 16, actuator_id 13
    "robot0:LFJ1",  # joint_id 17, actuator_id 14, tendon "LFT1", coupled joint
    "robot0:LFJ0",  # joint_id 18, actuator_id 14, tendon "LFT1", coupled joint
    "robot0:THJ4",  # joint_id 19, actuator_id 15
    "robot0:THJ3",  # joint_id 20, actuator_id 16
    "robot0:THJ2",  # joint_id 21, actuator_id 17
    "robot0:THJ1",  # joint_id 22, actuator_id 18
    "robot0:THJ0",  # joint_id 23, actuator_id 19
]

DEFAULT_INITIAL_QPOS = {
    "robot0:WRJ1": -0.16514339750464327,
    "robot0:WRJ0": -0.31973286565062153,
    "robot0:FFJ3": 0.14340512546557435,
    "robot0:FFJ2": 0.32028208333591573,
    "robot0:FFJ1": 0.7126053607727917,
    "robot0:FFJ0": 0.6705281001412586,
    "robot0:MFJ3": 0.000246444303701037,
    "robot0:MFJ2": 0.3152655251085491,
    "robot0:MFJ1": 0.7659800313729842,
    "robot0:MFJ0": 0.7323156897425923,
    "robot0:RFJ3": 0.00038520700007378114,
    "robot0:RFJ2": 0.36743546201985233,
    "robot0:RFJ1": 0.7119514095008576,
    "robot0:RFJ0": 0.6699446327514138,
    "robot0:LFJ4": 0.0525442258033891,
    "robot0:LFJ3": -0.13615534724474673,
    "robot0:LFJ2": 0.39872030433433003,
    "robot0:LFJ1": 0.7415570009679252,
    "robot0:LFJ0": 0.704096378652974,
    "robot0:THJ4": 0.003673823825070126,
    "robot0:THJ3": 0.5506291436028695,
    "robot0:THJ2": -0.014515151997119306,
    "robot0:THJ1": -0.0015229223564485414,
    "robot0:THJ0": -0.7894883021600622,
}

ACTUATOR_CTRLRANGE = {
    "robot0:A_WRJ1": [-0.4887, 0.1396],  # DEGREES (-28, 8)
    "robot0:A_WRJ0": [-0.6981, 0.4887],  # DEGREES (-40, 28)
    "robot0:A_FFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_FFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_FFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_MFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_MFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_MFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_RFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_RFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_RFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_LFJ4": [0.0, 0.7854],  # DEGREES (0, 45)
    "robot0:A_LFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "robot0:A_LFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "robot0:A_LFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "robot0:A_THJ4": [-1.0472, 1.0472],  # DEGREES (-60, 60)
    "robot0:A_THJ3": [0.0, 1.2217],  # DEGREES (0, 70)
    "robot0:A_THJ2": [-0.2094, 0.2094],  # DEGREES (-12, 12)
    "robot0:A_THJ1": [-0.5236, 0.5236],  # DEGREES (-30, 30)
    "robot0:A_THJ0": [-1.5708, 0.0],  # DEGREES (-90, 0)
}

ACTUATOR_JOINT_MAPPING: Dict[str, List[str]] = {
    "robot0:A_WRJ1": ["robot0:WRJ1"],
    "robot0:A_WRJ0": ["robot0:WRJ0"],
    "robot0:A_FFJ3": ["robot0:FFJ3"],
    "robot0:A_FFJ2": ["robot0:FFJ2"],
    "robot0:A_FFJ1": ["robot0:FFJ1", "robot0:FFJ0"],  # Coupled joints
    "robot0:A_MFJ3": ["robot0:MFJ3"],
    "robot0:A_MFJ2": ["robot0:MFJ2"],
    "robot0:A_MFJ1": ["robot0:MFJ1", "robot0:MFJ0"],  # Coupled joints
    "robot0:A_RFJ3": ["robot0:RFJ3"],
    "robot0:A_RFJ2": ["robot0:RFJ2"],
    "robot0:A_RFJ1": ["robot0:RFJ1", "robot0:RFJ0"],  # Coupled joints
    "robot0:A_LFJ4": ["robot0:LFJ4"],
    "robot0:A_LFJ3": ["robot0:LFJ3"],
    "robot0:A_LFJ2": ["robot0:LFJ2"],
    "robot0:A_LFJ1": ["robot0:LFJ1", "robot0:LFJ0"],  # Coupled joints
    "robot0:A_THJ4": ["robot0:THJ4"],
    "robot0:A_THJ3": ["robot0:THJ3"],
    "robot0:A_THJ2": ["robot0:THJ2"],
    "robot0:A_THJ1": ["robot0:THJ1"],
    "robot0:A_THJ0": ["robot0:THJ0"],
}

FINGER_GEOM_MAPPING: Dict[str, List[int]] = {
    "PM": [0, 1, 2],
    "FF": [3, 4, 5],
    "MF": [6, 7, 8],
    "RF": [9, 10, 11],
    "LF": [12, 13, 14, 15],
    "TH": [16, 17, 18],
}

JOINT_LIMITS: Dict[str, np.ndarray] = {}
for actuator, ctrlrange in ACTUATOR_CTRLRANGE.items():
    joints = ACTUATOR_JOINT_MAPPING[actuator]
    for joint in joints:
        JOINT_LIMITS[joint] = np.array(ctrlrange) / len(joints)

MODES = {
    "face_up": [-np.pi / 2, 0, 0],
    "face_down": [np.pi / 2, 0, np.pi]
}

def get_actuator_mapping():
    def find(a, x):
        for idx, i in enumerate(a):
            if i == x:
                return idx
        raise Exception

    actuator_mapping = [None for i in JOINTS]
    for actuator in ACTUATOR_JOINT_MAPPING:
        for joint in ACTUATOR_JOINT_MAPPING[actuator]:
            actuator_mapping[find(JOINTS, joint)] = find(ACTUATOR_JOINT_MAPPING, actuator)

    return actuator_mapping


def get_template(mjcf_xml):
    # no recursion - only one expansion is needed
    default_template = dict()
    for class_tempalte in mjcf_xml.root_element.find("default").findall("default"):

        template_name = class_tempalte.attrib.get("class", "")
        assert template_name, "class name cannot be empty!"

        default_template[template_name] = x_dict = dict()
        for body_attrib_template in class_tempalte:
            attrib_name = body_attrib_template.tag
            x_dict[attrib_name] = body_attrib_template.attrib

    return default_template


def combine_str(robot_name, part_name):
    return f"{robot_name}:{part_name}"

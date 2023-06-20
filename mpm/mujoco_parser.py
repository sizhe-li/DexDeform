import os.path
import xml.etree.ElementTree as et
from typing import Any, Tuple
from typing import Optional

from mpm.robots.interface import *

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
XML_DIR = os.path.join(ASSETS_DIR, 'robots')

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1):
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_identity():
    return np.array([1, 0, 0, 0])


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape[-1] == 3
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.reshape(angle, axis[..., :1].shape)
    w = np.cos(angle / 2.0)
    v = np.sin(angle / 2.0) * axis
    quat = np.concatenate([w, v], axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    assert np.array_equal(quat.shape[:-1], axis.shape[:-1])
    return quat


def get_mesh_data(rt: et.Element, name):
    asset_dir = os.path.join(ASSETS_DIR, "stls", "hand")
    for x in rt.findall(".//mesh"):
        if x.attrib.get("name", "") == name:
            mesh_file = x.attrib.get("file", "")
            mesh_file = os.path.join(asset_dir, mesh_file)

            scale = x.attrib.get("scale", "")

            # NOTE: ignore do not load mesh
            # assert os.path.exists(mesh_file)

            return mesh_file, scale


# credit: robogym
class MujocoXML:
    """
    Class that combines multiple MuJoCo XML files into a single one.
    """

    ###############################################################################################
    # CONSTRUCTION
    @classmethod
    def parse(cls, xml_filename: str, use_template=False):
        """ Parse given xml filename into the MujocoXML model """

        xml_full_path = os.path.join(XML_DIR, xml_filename)
        if not os.path.exists(xml_full_path):
            raise Exception(xml_full_path)

        with open(xml_full_path) as f:
            xml_root = et.parse(f).getroot()

        xml = cls(xml_root)
        xml.load_includes(os.path.dirname(os.path.abspath(xml_full_path)))
        if use_template:
            xml.apply_default_template()

        return xml

    @classmethod
    def from_string(cls, contents: str):
        """ Construct MujocoXML from string """
        xml_root = et.XML(contents)
        xml = cls(xml_root)
        xml.load_includes()
        return xml

    def __init__(self, root_element: Optional[et.Element] = None):
        """ Create new MujocoXML class """
        # This is the root element of the XML document we'll be modifying
        if root_element is None:
            # Create empty root element
            self.root_element = et.Element("mujoco")
        else:
            # Initialize it from the existing thing
            self.root_element = root_element

    def xml_string(self):
        """ Return combined XML as a string """
        return et.tostring(self.root_element, encoding="unicode", method="xml")

    def load_includes(self, include_root=""):
        """
        Some mujoco files contain includes that need to be process on our side of the system
        Find all elements that have an 'include' child
        """
        for element in self.root_element.findall(".//include/.."):
            # Remove in a second pass to avoid modifying list while iterating it
            elements_to_remove_insert = []

            for idx, subelement in enumerate(element):
                if subelement.tag == "include":
                    # Branch off initial filename
                    include_path = os.path.join(include_root, subelement.get("file"))

                    include_element = MujocoXML.parse(include_path)

                    elements_to_remove_insert.append(
                        (idx, subelement, include_element.root_element)
                    )

            # Iterate in reversed order to make sure indices are not screwed up
            for idx, to_remove, to_insert in reversed(elements_to_remove_insert):
                element.remove(to_remove)
                to_insert_list = list(to_insert)

                # Insert multiple elements
                for i in range(len(to_insert)):
                    element.insert(idx + i, to_insert_list[i])

        return self

    def apply_default_template(self):
        xml_root = self.root_element

        def get_template(rt: et.Element):
            # no recursion - only one expansion is needed
            default_template = dict()
            for class_template in rt.find("default").findall("default"):

                template_name = class_template.attrib.get("class", "")
                assert template_name, "class name cannot be empty!"

                default_template[template_name] = x_dict = dict()
                for body_attrib_template in class_template:
                    attrib_name = body_attrib_template.tag
                    x_dict[attrib_name] = body_attrib_template.attrib

            return default_template

        default_template = get_template(xml_root)

        def apply_template_to_attribs(template, x):
            for key in template:
                if key not in x.attrib.keys():
                    # apply k, v from template if k does not exist
                    x.set(key, template[key])

        def traverse(rt: et.Element):
            for x in rt:
                x_attribs = x.attrib
                if "class" in x_attribs:
                    template_name = x_attribs.get("class", "")
                    template = default_template[template_name]

                    apply_template_to_attribs(template[x.tag], x)


                elif "childclass" in x_attribs:

                    template_name = x_attribs.get("childclass", "")
                    template = default_template[template_name]

                    def apply_child_template(rt: et.Element):

                        for x in rt:
                            if x.tag in template:
                                apply_template_to_attribs(template[x.tag], x)

                            apply_child_template(x)

                traverse(x)

        rt = xml_root.find("worldbody")
        traverse(rt)


def homogeneous_matrix_from_pos_mat_np(pos, mat):
    m = np.eye(4)
    m[:3, :3] = mat
    m[:3, 3] = pos
    return m


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape[-1] == 3
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.reshape(angle, axis[..., :1].shape)
    w = np.cos(angle / 2.0)
    v = np.sin(angle / 2.0) * axis
    quat = np.concatenate([w, v], axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    assert np.array_equal(quat.shape[:-1], axis.shape[:-1])
    return quat


def get_joint_matrix(pos, angle, axis):
    def transform_rot_x_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the X axis by given angle in radians
        """
        m = np.eye(4)
        m[1, 1] = m[2, 2] = np.cos(angle)
        s = np.sin(angle)
        m[1, 2] = -s
        m[2, 1] = s
        m[:3, 3] = pos
        return m

    def transform_rot_y_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the Y axis by given angle in radians
        """
        m = np.eye(4)
        m[0, 0] = m[2, 2] = np.cos(angle)
        s = np.sin(angle)
        m[0, 2] = s
        m[2, 0] = -s
        m[:3, 3] = pos
        return m

    def transform_rot_z_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the Z axis by given angle in radians
        """
        m = np.eye(4)
        m[0, 0] = m[1, 1] = np.cos(angle)
        s = np.sin(angle)
        m[0, 1] = -s
        m[1, 0] = s
        m[:3, 3] = pos
        return m

    if abs(axis[0]) == 1.0 and axis[1] == 0.0 and axis[2] == 0.0:
        return transform_rot_x_matrix(pos, angle * axis[0])
    elif axis[0] == 0.0 and abs(axis[1]) == 1.0 and axis[2] == 0.0:
        return transform_rot_y_matrix(pos, angle * axis[1])
    elif axis[0] == 0.0 and axis[1] == 0.0 and abs(axis[2]) == 1.0:
        return transform_rot_z_matrix(pos, angle * axis[2])
    else:
        return homogeneous_matrix_from_pos_mat_np(
            pos, quat2mat(quat_from_angle_and_axis(angle, axis))
        )


class MujocoGeom:
    def __init__(self,
                 parent_name: str,
                 matrix: np.array,
                 is_mesh: bool,
                 data,
                 ):
        self.parent_name = parent_name
        self.matrix = matrix
        self.is_mesh = is_mesh
        self.data = data

        # if mesh then we save the name
        # if not mesh then we save the type and size in a tuple


def prepare(filename, scale=1.0):
    mxml: MujocoXML = MujocoXML.parse(filename, use_template=True)
    root_body_name: str = "robot0:hand mount"
    root_body_pos: np.array = np.array([1.0, 1.25, 0.15])
    root_body_euler: np.array = np.array([np.pi / 2, 0, np.pi])
    target_sites: List[str] = FINGERTIP_SITE_NAMES
    joint_names: List[str] = JOINTS

    IDENTITY_QUAT = quat_identity()
    ROOT_BODY_PARENT = "NONE"

    target_sites_idx: Dict[str, int] = {
        v: idx for idx, v in enumerate(target_sites)
    }

    joint_names_idx: Dict[str, int] = {v: idx for idx, v in enumerate(joint_names)}
    num_sites = len(target_sites)
    site_info: List[Optional[Tuple]] = [None] * num_sites  # (4d matrix, parentBody)
    joint_info: List[Optional[Tuple]] = [None] * len(joint_names)  # (axis, pos)

    body_info: Dict[
        str, Any
    ] = dict()  # name => (4d homegeneous matrix, parentbody)
    body_joints: Dict[str, str] = dict()  # body => joints

    def get_matrix(x: et.Element):
        pos = np.fromstring(x.attrib.get("pos"), sep=" ") * scale
        if "euler" in x.attrib:
            euler = np.fromstring(x.attrib.get("euler"), sep=" ")
            return homogeneous_matrix_from_pos_mat_np(pos, euler2mat(euler))
        elif "axisangle" in x.attrib:
            axis_angle = np.fromstring(x.attrib.get("axisangle"), sep=" ")
            quat = quat_from_angle_and_axis(
                axis_angle[-1], np.array(axis_angle[:-1])
            )
            return homogeneous_matrix_from_pos_mat_np(pos, quat2mat(quat))
        elif "quat" in x.attrib:
            quat = np.fromstring(x.attrib.get("quat"), sep=" ")
            return homogeneous_matrix_from_pos_mat_np(pos, quat2mat(quat))
        else:
            quat = IDENTITY_QUAT
            return homogeneous_matrix_from_pos_mat_np(pos, quat2mat(quat))

    vis_geoms: List[MujocoGeom] = []
    col_geoms: List[MujocoGeom] = []

    def traverse(rt: et.Element, parent_body: str):
        assert rt.tag == "body", "only start from body tag in xml"
        matrix = get_matrix(rt)
        body_name = rt.attrib.get("name", "noname_body_%d" % len(body_info))
        body_info[body_name] = (matrix, parent_body)

        # parse joint
        x = rt.find("joint")
        joint_name = None
        if x is not None:
            joint_name = x.attrib.get("name", "")
            joint_idx: int = joint_names_idx.get(joint_name, -1)
            if joint_idx != -1:  # if in our target joints
                assert (
                        x.attrib.get("type", "hinge") == "hinge"
                ), "currently only support hinge joints"

                pos = np.fromstring(x.attrib.get("pos"), sep=" ")
                axis = np.fromstring(x.attrib.get("axis"), sep=" ")

                joint_info[joint_idx] = (pos, axis)

                assert (
                        joint_name not in body_joints
                ), "Only support open chain system, unsupported rigid bodies"
                body_joints[body_name] = joint_name

        # parse geometry
        for x in rt.findall("geom"):
            # get parent name
            parent_name = joint_name if joint_name is not None else body_name

            # get homogeneous matrix
            if not x.attrib.get("pos", ""):
                matrix = np.eye(4)
            else:
                matrix = get_matrix(x)

            # load data depending on whether it is mesh
            mesh_name = x.attrib.get("mesh", "")
            is_mesh = len(mesh_name) > 0
            if is_mesh:
                mesh_path, mesh_scale = get_mesh_data(mxml.root_element, mesh_name)
                data = {"path": mesh_path, "scale": mesh_scale}
            else:
                data = {k: x.attrib.get(k, "") for k in ("type", "size")}
                size = [float(x) * scale for x in data["size"].split(" ")]
                data["size"] = size

            geom = MujocoGeom(parent_name, matrix, is_mesh, data)

            # determine group
            geom_group = x.attrib.get("group", "")
            if geom_group == "1":
                vis_geoms.append(geom)
            elif geom_group == "4":
                col_geoms.append(geom)

        for x in rt.findall("site"):
            site_idx = target_sites_idx.get(x.attrib.get("name", ""), -1)
            if site_idx != -1:
                matrix = get_matrix(x)
                site_info[site_idx] = (matrix, body_name)

        for x in rt.findall("body"):
            traverse(x, body_name)

    #######################
    ### Begin traversal ###
    #######################

    rt = None
    for child in mxml.root_element.find("worldbody").findall("body"):  # type: ignore
        if child.attrib.get("name", "") == root_body_name:
            rt = child
            break

    assert rt is not None, "no root body found in xml"
    traverse(rt, ROOT_BODY_PARENT)

    ##########################
    # Build Computation Flow #
    ##########################

    root_matrix = homogeneous_matrix_from_pos_mat_np(
        root_body_pos, euler2mat(root_body_euler)
    )

    site_computations_inds = [[] for i in range(num_sites)]
    site_computations_mats = [[] for i in range(num_sites)]

    for i in range(num_sites):
        (matrix, parent_body) = site_info[i]
        while parent_body != ROOT_BODY_PARENT:
            parent_matrix, new_parent_body = body_info[parent_body]
            joint_name = body_joints.get(parent_body, "")

            # certain body has joint, certain doesn't
            if joint_name:
                site_computations_inds[i].append(("body_idx", len(site_computations_mats[i])))
                site_computations_mats[i].append(matrix)
                site_computations_inds[i].append(("joint_idx", joint_names_idx[joint_name]))
                matrix = parent_matrix
            else:
                matrix = parent_matrix @ matrix
            parent_body = new_parent_body

    site_computations = {
        "mats": site_computations_mats,
        "inds": [list(reversed(x)) for x in site_computations_inds],
    }

    def body_to_matrix(body_name):
        rt = None
        for child in mxml.root_element.find("worldbody").findall(".//body"):
            if child.attrib.get("name", "") == body_name:
                rt = child
                break
        return get_matrix(rt)

    root_info = {"root_matrix": root_matrix}
    for name in ["robot0:hand mount", "robot0:forearm", "robot0:wrist"]:
        root_info[name] = body_to_matrix(name)

    geometries: Dict[str, MujocoGeom] = {"vis": vis_geoms, "col": col_geoms}

    return root_info, site_computations, joint_info, geometries

import sys

import numpy as np
from stl import mesh
import math
import logging
from svgelements import SVG, Circle, Use, Rect, Group, Path
from typing import List, Dict, Any
from pygerber.gerberx3.api.v2 import GerberFile
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SimpleMeshUnionProcessor:
    """
    Упрощенный класс для объединения мешей без trimesh
    """

    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance

    def union_intersecting_meshes(self, mesh_list):
        """
        Основной публичный метод для объединения пересекающихся мешей
        """
        if len(mesh_list) <= 1:
            return mesh_list.copy()

        # Находим группы пересекающихся мешей
        groups = self._find_connected_groups(mesh_list)

        # Объединяем меши в группах
        result = []
        for group in groups:
            if len(group) == 1:
                result.append(mesh_list[group[0]])
            else:
                united = self._unite_meshes([mesh_list[i] for i in group])
                result.append(united)

        return result

    def _find_connected_groups(self, mesh_list):
        """Находит группы связанных мешей"""
        groups = []
        visited = set()

        for i in range(len(mesh_list)):
            if i not in visited:
                group = self._find_connected_group(i, mesh_list, visited)
                groups.append(group)

        return groups

    def _find_connected_group(self, start_idx, mesh_list, visited):
        """Находит группу связанных мешей"""
        group = []
        stack = [start_idx]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                group.append(current)

                # Ищем соседей
                for i in range(len(mesh_list)):
                    if i not in visited and self._check_intersection_simple(
                            mesh_list[current], mesh_list[i]):
                        stack.append(i)

        return group

    def _check_intersection_simple(self, mesh1, mesh2):
        """Простая проверка пересечения через bounding boxes"""
        min1, max1 = mesh1.min_, mesh1.max_
        min2, max2 = mesh2.min_, mesh2.max_

        # Проверяем пересечение bbox с допуском
        for axis in range(3):
            if (max1[axis] + self.tolerance < min2[axis] or
                    min1[axis] - self.tolerance > max2[axis]):
                return False
        return True

    def _unite_meshes(self, meshes):
        """Объединяет меши через конкатенацию"""
        all_vectors = []

        for m in meshes:
            all_vectors.append(m.vectors)

        # Конкатенируем все векторы
        united_vectors = np.vstack(all_vectors)

        # Создаем новый меш
        data = np.zeros(len(united_vectors), dtype=mesh.Mesh.dtype)
        data['vectors'] = united_vectors

        return mesh.Mesh(data)


class PolygonTriangulator:
    """Класс для триангуляции полигонов"""

    @staticmethod
    def ear_clip_triangulation(points: np.ndarray) -> List[List[int]]:
        """
        Триангуляция полигона методом ear clipping
        """
        if len(points) < 3:
            return []

        # Проверяем направление полигона (должно быть против часовой стрелки)
        if not PolygonTriangulator.is_counter_clockwise(points):
            points = points[::-1]

        indices = list(range(len(points)))
        triangles = []

        while len(indices) > 3:
            n = len(indices)
            found_ear = False

            for i in range(n):
                prev_idx = indices[i - 1]
                curr_idx = indices[i]
                next_idx = indices[(i + 1) % n]

                prev_point = points[prev_idx]
                curr_point = points[curr_idx]
                next_point = points[next_idx]

                # Проверяем, является ли вершина ухом
                if PolygonTriangulator.is_ear(prev_point, curr_point, next_point, points, indices):
                    triangles.append([prev_idx, curr_idx, next_idx])
                    indices.pop(i)
                    found_ear = True
                    break

            if not found_ear:
                # Fallback: используем выпуклую оболочку
                return PolygonTriangulator.convex_hull_triangulation(points)

        if len(indices) == 3:
            triangles.append(indices)

        return triangles

    @staticmethod
    def is_counter_clockwise(points: np.ndarray) -> bool:
        """Проверяет, идет ли полигон против часовой стрелки"""
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return area > 0

    @staticmethod
    def is_ear(a: np.ndarray, b: np.ndarray, c: np.ndarray,
               points: np.ndarray, indices: List[int]) -> bool:
        """Проверяет, является ли треугольник ухом"""
        # Проверяем выпуклость
        ab = b - a
        bc = c - b
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        if cross <= 0:
            return False

        # Проверяем, что внутри треугольника нет других вершин
        triangle = np.array([a, b, c])
        for idx in indices:
            point = points[idx]
            if np.array_equal(point, a) or np.array_equal(point, b) or np.array_equal(point, c):
                continue
            if PolygonTriangulator.point_in_triangle(point, triangle):
                return False

        return True

    @staticmethod
    def point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
        """Проверяет, находится ли точка внутри треугольника"""

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(point, triangle[0], triangle[1])
        d2 = sign(point, triangle[1], triangle[2])
        d3 = sign(point, triangle[2], triangle[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    @staticmethod
    def convex_hull_triangulation(points: np.ndarray) -> List[List[int]]:
        """Триангуляция через выпуклую оболочку (fallback)"""
        try:
            hull = ConvexHull(points)
            triangles = []
            for simplex in hull.simplices:
                triangles.append(list(simplex))
            return triangles
        except:
            # Простая триангуляция от центральной точки
            center = np.mean(points, axis=0)
            center_idx = len(points)
            all_points = np.vstack([points, center])

            triangles = []
            n = len(points)
            for i in range(n):
                triangles.append([i, (i + 1) % n, center_idx])

            return triangles


class SVGToSTL:
    def __init__(self, extrusion_height=1.0, tolerance=1.0, circle_segments=32, start_position=[0, 0, 0]):
        self.height = extrusion_height
        self.tolerance = tolerance
        self.circle_segments_count = circle_segments
        self.logger = logging.getLogger(__name__)
        self.triangulator = PolygonTriangulator()
        self.user_offset = start_position
        self.offset = [float,float,float]
        self.jig = {
            'start_position': [0,0,0],
            'x_len': 50,
            'y_len': 50,
            'wall_width': 5,
            'wall_height': 1.5,
        }
        logging.info(f"Initializing offset: {self.user_offset}")

    def parse_svg(self, svg_file):
        """Парсит SVG и извлекает информацию о смещении"""
        data = []

        # Парсим SVG
        svg = SVG.parse(svg_file)

        # Получаем размеры SVG для вычисления смещения
        width = 0
        height = 0

        if hasattr(svg, 'width') and svg.width is not None:
            width = float(svg.width)

        if hasattr(svg, 'height') and svg.height is not None:
            height = float(svg.height)
        offset = [0,0,0]
        offset[0] = float(self.user_offset[0]) # + width/2)
        offset[1] = float(self.user_offset[1]) # + height/2)
        offset[2] = float(self.user_offset[2])
        logging.info(f"Update offset: {offset}")

        # Вычисляем смещение: чтобы центр [width/2, height/2] стал [0, 0]
        # Тогда левый нижний угол будет в [0, 0] после преобразования
        self.offset = np.array(offset, dtype=np.float64)

        # Обрабатываем элементы
        for element in svg.elements():
            _el_data = {}
            if isinstance(element, Circle):
                _el_data = {
                    'type': 'circle',
                    'cx': element.cx,
                    'cy': element.cy,
                    'r': element.implicit_r
                }
            elif isinstance(element, Rect):
                _el_data = {
                    'type': 'rect',
                    'x': element.x,
                    'y': element.y,
                    'width': element.width,
                    'height': element.height,
                }
            elif isinstance(element, Path):
                points = []
                try:
                    for s in element.segments():
                        if hasattr(s, 'start') and s.start is not None:
                            points.append((s.start.x, s.start.y))
                    if len(points) >= 3:
                        _el_data = {
                            'type': 'polygon',
                            'points': points
                        }
                except Exception as e:
                    self.logger.warning(f"Error parsing Path: {e}")

            elif isinstance(element, Use):
                if len(element) > 0 and len(element[0]) > 0:
                    _element = element[0][0]
                    if isinstance(_element, Circle):
                        _el_data = {
                            'type': 'circle',
                            'cx': _element.cx,
                            'cy': _element.cy,
                            'r': _element.implicit_r
                        }
                    elif isinstance(_element, Rect):
                        _el_data = {
                            'type': 'rect',
                            'x': _element.x,
                            'y': _element.y,
                            'width': _element.width,
                            'height': _element.height,
                        }

            elif isinstance(element, Group):
                for sub_element in element:
                    sub_data = self._extract_element_data(sub_element)
                    if sub_data:
                        data.append(sub_data)

            if _el_data:
                self.logger.debug(f'svg element: {_el_data["type"]} {_el_data}')
                data.append(_el_data)

        return data

    def generate_mesh_jig(self, start_position=[0, 0, 0], x_len=100, y_len=100, wall_width=5, wall_height=1.4):
        """
        Генерирует настоящий L-образный кондуктор (две перпендикулярные стенки)
        """
        inner_x, inner_y, inner_z = start_position

        # Вершины для L-образного кондуктора
        vertices = [
            # Вертикальная стенка (X направление)
            [inner_x, inner_y, inner_z],  # 0
            [inner_x + wall_width, inner_y, inner_z],  # 1
            [inner_x + wall_width, inner_y + y_len, inner_z],  # 2
            [inner_x, inner_y + y_len, inner_z],  # 3
            [inner_x, inner_y, inner_z + wall_height],  # 4
            [inner_x + wall_width, inner_y, inner_z + wall_height],  # 5
            [inner_x + wall_width, inner_y + y_len, inner_z + wall_height],  # 6
            [inner_x, inner_y + y_len, inner_z + wall_height],  # 7

            # Горизонтальная стенка (Y направление)
            [inner_x, inner_y, inner_z],  # 8 (дубликат 0)
            [inner_x + x_len, inner_y, inner_z],  # 9
            [inner_x + x_len, inner_y + wall_width, inner_z],  # 10
            [inner_x, inner_y + wall_width, inner_z],  # 11
            [inner_x, inner_y, inner_z + wall_height],  # 12 (дубликат 4)
            [inner_x + x_len, inner_y, inner_z + wall_height],  # 13
            [inner_x + x_len, inner_y + wall_width, inner_z + wall_height],  # 14
            [inner_x, inner_y + wall_width, inner_z + wall_height]  # 15
        ]

        # Грани для вертикальной стенки
        vertical_faces = [
            [0, 1, 5], [0, 5, 4],  # перед
            [3, 6, 2], [3, 7, 6],  # зад
            [1, 2, 6], [1, 6, 5],  # внешняя
            [0, 7, 3], [0, 4, 7],  # внутренняя
            [0, 3, 2], [0, 2, 1],  # верх
            [4, 5, 6], [4, 6, 7]  # низ
        ]

        # Грани для горизонтальной стенки
        horizontal_faces = [
            [8, 9, 13], [8, 13, 12],  # перед
            [11, 14, 10], [11, 15, 14],  # зад
            [9, 10, 14], [9, 14, 13],  # внешняя
            [8, 15, 11], [8, 12, 15],  # внутренняя
            [8, 11, 10], [8, 10, 9],  # верх
            [12, 13, 14], [12, 14, 15]  # низ
        ]

        all_faces = vertical_faces + horizontal_faces

        vertices_np = np.array(vertices)
        faces_np = np.array(all_faces)

        mesh_data = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces_np):
            mesh_data.vectors[i] = vertices_np[face]

        return mesh_data

    def _extract_element_data(self, element):
        """Извлекает данные из элемента"""
        if isinstance(element, Circle):
            return {
                'type': 'circle',
                'cx': element.cx,
                'cy': element.cy,
                'r': element.implicit_r
            }
        elif isinstance(element, Rect):
            return {
                'type': 'rect',
                'x': element.x,
                'y': element.y,
                'width': element.width,
                'height': element.height,
            }
        elif isinstance(element, Path):
            points = []
            try:
                for s in element.segments():
                    if hasattr(s, 'start') and s.start is not None:
                        points.append((s.start.x, s.start.y))
                if len(points) >= 3:
                    return {
                        'type': 'polygon',
                        'points': points
                    }
            except:
                pass
        return None

    def _apply_offset(self, points: np.ndarray) -> np.ndarray:
        """Применяет смещение к точкам"""
        offset_points = points.copy()

        # Преобразуем координаты:
        # от системы с центром в [width/2, height/2]
        # к системе с левым нижним углом в [0, 0]
        offset_points[:, 0] = offset_points[:, 0] + self.offset[0]
        offset_points[:, 1] = offset_points[:, 1] + self.offset[1]
        offset_points[:, 2] = offset_points[:, 2] + self.offset[2]

        return offset_points

    def generate_mesh_for_element(self, element: Dict[str, Any]) -> mesh.Mesh:
        """
        Генерирует STL mesh для элемента с заданной высотой
        """
        if element['type'] == 'polygon':
            return self.generate_polygon_mesh(element, self.height)
        elif element['type'] == 'circle':
            return self.generate_circle_mesh(element, self.height)
        elif element['type'] == 'rect':
            return self.generate_rect_mesh(element, self.height)
        else:
            self.logger.warning(f"Unknown element type: {element['type']}")
            exit(1)

    def generate_polygon_mesh(self, element: Dict[str, Any], height: float) -> mesh.Mesh:
        """Генерирует mesh для полигона с правильной триангуляцией"""
        points = np.array(element['points'], dtype=np.float64)
        n = len(points)

        if n < 3:
            self.logger.warning(f"Polygon with only {n} points, skipping")
            return None

        # Проверяем и исправляем порядок вершин (должен быть против часовой стрелки)
        if not self.is_counter_clockwise(points):
            points = points[::-1]

        # Для прямоугольников используем специальную триангуляцию
        if n == 4 and self.is_rectangle(points):
            triangles = [[0, 1, 2], [0, 2, 3]]
        else:
            # Для других полигонов используем ear clipping
            try:
                triangles = self.triangulator.ear_clip_triangulation(points)
                if not triangles:
                    self.logger.warning("Failed to triangulate polygon")
                    return None
            except Exception as e:
                self.logger.error(f"Triangulation error: {e}")
                return None

        # Создаем вершины для нижней и верхней граней
        bottom_vertices = np.hstack([points, np.zeros((n, 1))])
        top_vertices = np.hstack([points, np.full((n, 1), height)])

        # Все вершины: сначала нижние, потом верхние
        vertices = np.vstack([bottom_vertices, top_vertices])

        # Применяем смещение ко всем вершинам
        vertices = self._apply_offset(vertices)

        # Создаем грани
        faces = []

        # Нижняя грань
        for triangle in triangles:
            faces.append(triangle)

        # Верхняя грань (инвертированная триангуляция)
        for triangle in triangles:
            inverted_triangle = [v + n for v in triangle[::-1]]
            faces.append(inverted_triangle)

        # Боковые грани
        for i in range(n):
            next_i = (i + 1) % n
            bottom_i = i
            bottom_next = next_i
            top_i = i + n
            top_next = next_i + n

            faces.append([bottom_i, bottom_next, top_i])
            faces.append([bottom_next, top_next, top_i])

        return self.create_mesh_from_faces(vertices, faces)

    def generate_circle_mesh(self, element: Dict[str, Any], height: float) -> mesh.Mesh:
        """Генерирует mesh для круга"""
        cx, cy = element['cx'], element['cy']
        r = element['r']
        segments = self.circle_segments_count

        # Создаем точки окружности
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)

        bottom_vertices = []
        top_vertices = []

        for angle in angles:
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            bottom_vertices.append([x, y, 0])
            top_vertices.append([x, y, height])

        # Центральные точки
        bottom_center = [cx, cy, 0]
        top_center = [cx, cy, height]

        # Все вершины
        all_vertices = [bottom_center] + bottom_vertices + [top_center] + top_vertices
        vertices = np.array(all_vertices)

        # Применяем смещение ко всем вершинам
        vertices = self._apply_offset(vertices)

        # Создаем грани
        faces = []
        n = segments

        # Нижний круг
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([0, i + 1, next_i + 1])

        # Верхний круг (инвертированный)
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([n + 1, n + 2 + next_i, n + 2 + i])

        # Боковые грани
        for i in range(n):
            next_i = (i + 1) % n
            bottom_i = i + 1
            bottom_next = next_i + 1
            top_i = n + 2 + i
            top_next = n + 2 + next_i

            faces.append([bottom_i, bottom_next, top_i])
            faces.append([bottom_next, top_next, top_i])

        return self.create_mesh_from_faces(vertices, faces)

    def generate_rect_mesh(self, element: Dict[str, Any], height: float) -> mesh.Mesh:
        """Генерирует mesh для прямоугольника"""
        x, y = element['x'], element['y']
        width, height_rect = element['width'], element['height']

        # Вершины прямоугольника
        bottom_vertices = np.array([
            [x, y, 0],
            [x + width, y, 0],
            [x + width, y + height_rect, 0],
            [x, y + height_rect, 0]
        ])

        top_vertices = bottom_vertices.copy()
        top_vertices[:, 2] = height

        vertices = np.vstack([bottom_vertices, top_vertices])

        # Применяем смещение ко всем вершинам
        vertices = self._apply_offset(vertices)

        # Грани
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 1], [1, 4, 5],
            [1, 5, 2], [2, 5, 6],
            [2, 6, 3], [3, 6, 7],
            [3, 7, 0], [0, 7, 4]
        ]

        return self.create_mesh_from_faces(vertices, faces)

    def is_counter_clockwise(self, points: np.ndarray) -> bool:
        """Проверяет, идет ли полигон против часовой стрелки"""
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return area > 0

    def is_rectangle(self, points: np.ndarray, tolerance=1e-6) -> bool:
        """Проверяет, является ли полигон прямоугольником"""
        if len(points) != 4:
            return False

        # Проверяем углы (должны быть 90 градусов)
        vectors = []
        for i in range(4):
            v = points[(i + 1) % 4] - points[i]
            vectors.append(v)

        # Проверяем перпендикулярность соседних сторон
        for i in range(4):
            dot = np.dot(vectors[i], vectors[(i + 1) % 4])
            if abs(dot) > tolerance:
                return False

        return True

    @staticmethod
    def create_mesh_from_faces(vertices: np.ndarray, faces: List[List[int]]) -> mesh.Mesh:
        """Создает STL mesh из вершин и граней"""
        # Создаем mesh
        data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)

        for i, face in enumerate(faces):
            # Получаем вершины для этой грани
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]

            # Вычисляем нормаль
            normal = np.cross(v2 - v1, v3 - v1)
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                normal = normal / normal_length
            else:
                normal = np.array([0, 0, 1])

            # Заполняем данные
            data['vectors'][i] = np.array([v1, v2, v3])
            data['normals'][i] = normal

        return mesh.Mesh(data)

    def merge_meshes(self, meshes: List[mesh.Mesh]) -> mesh.Mesh:
        """Объединяет несколько мешей в один"""
        if not meshes:
            return mesh.Mesh(np.array([], dtype=mesh.Mesh.dtype))

        # Фильтруем None значения
        valid_meshes = [m for m in meshes if m is not None and len(m.data) > 0]

        if not valid_meshes:
            return mesh.Mesh(np.array([], dtype=mesh.Mesh.dtype))

        # Объединяем все данные
        combined_data = np.concatenate([m.data for m in valid_meshes])
        return mesh.Mesh(combined_data)

    def save_stl(self, mesh_data: mesh.Mesh, stl_path: str):
        """Сохраняет меш в STL файл"""
        try:
            mesh_data.save(stl_path)
            self.logger.info(f"STL file saved to: {stl_path}")

            if not mesh_data.is_closed():
                self.logger.warning("Mesh is not closed. This may cause issues with some 3D printing software.")
            else:
                self.logger.info("Mesh is closed.")

        except Exception as e:
            self.logger.error(f"Error saving STL file: {e}")

    def convert(self, f_in: str, file_out: str):
        """Основной метод конвертации SVG в STL"""
        self.logger.info(f"Converting {f_in} to {file_out}")

        # Парсим SVG
        elements = self.parse_svg(f_in)
        self.logger.info(f"Found {len(elements)} elements in SVG")

        # Генерируем меши
        meshes = []
        for i, element in enumerate(elements):
            self.logger.debug(f"Processing element {i}: {element['type']}")
            new_mesh = self.generate_mesh_for_element(element)
            if new_mesh is not None:
                meshes.append(new_mesh)

        if self.jig['wall_height'] > 0:
            jig_mesh = self.generate_mesh_jig(
                start_position=self.jig['start_position'],
                x_len=self.jig['x_len'],
                y_len=self.jig['y_len'],
                wall_width=self.jig['wall_width'],
                wall_height=self.jig['wall_height'],
            )
            meshes.append(jig_mesh)

        self.logger.info(f"Generated {len(meshes)} valid meshes")

        if not meshes:
            self.logger.warning("No valid meshes generated")
            return

        processor = SimpleMeshUnionProcessor(tolerance=1e-6)
        result_meshes = processor.union_intersecting_meshes(meshes)
        self.logger.info(f"Original: {len(meshes)} meshes, After union: {len(result_meshes)} meshes")

        # Объединяем меши
        result = self.merge_meshes(result_meshes)

        # Сохраняем результат
        self.save_stl(result, file_out)
        self.logger.info("Conversion completed successfully")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        gerber_file = sys.argv[1]
    else:
        gerber_file = 'board.GTL'
    pcb_height = 1.2
    border_width = 0.8
    svg_file = gerber_file.replace('.GTL', '.svg')
    stl_file = gerber_file.replace('.GTL', '.stl')
    GerberFile.from_file(gerber_file).parse().render_svg(destination=svg_file, scale=1)
    converter = SVGToSTL(extrusion_height=0.4, start_position=[border_width+1,border_width+1,pcb_height])
    converter.jig = {
            'start_position': [0,0,0],
            'x_len': 90,
            'y_len': 40,
            'wall_width': border_width,
            'wall_height': pcb_height,
        }
    converter.convert(f_in=svg_file, file_out=stl_file)

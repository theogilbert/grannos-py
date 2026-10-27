import re
from collections.abc import Awaitable, Callable, Mapping

import pytest

from grannos import diagram
from grannos.diagram import DiagramError, build_diagram
from grannos.diagram.place import PlaceResult, Spacing
from grannos.diagram.route import NoRouteError
from grannos.protocol import (
    DescribeResult,
    EntityDescription,
    FieldDescription,
    IndexDescription,
    TableReference,
)

Describe = Callable[[list[str]], Awaitable[DescribeResult]]


def _describe_from(table_by_path: Mapping[tuple[str, ...], DescribeResult]) -> Describe:
    async def describe(path: list[str]) -> DescribeResult:
        return table_by_path.get(tuple(path))

    return describe


def _entity(
    name: str,
    schema: str | None = None,
    fields: list[FieldDescription] | None = None,
) -> EntityDescription:
    return EntityDescription(
        name=name, kind="table", schema=schema, properties=fields or []
    )


def _field(
    name: str,
    type_: str = "INTEGER",
    pk: bool = False,
    outgoing: list[TableReference] | None = None,
    incoming: list[TableReference] | None = None,
) -> FieldDescription:
    return FieldDescription(
        name=name,
        types=[type_],
        pk=pk,
        outgoing_references=outgoing or [],
        incoming_references=incoming or [],
    )


def _outgoing(
    table: str, column: str, ref_table: str, ref_column: str, unique: bool = False
) -> TableReference:
    return TableReference(
        table=table,
        column=column,
        ref_table=ref_table,
        ref_column=ref_column,
        unique=unique,
    )


def _incoming(
    owner_table: str, owner_column: str, this_table: str, this_column: str
) -> TableReference:
    """A reference owned by *owner_table* that targets *this_table*'s field —
    i.e. what belongs on ``this_table``'s ``this_column`` field's
    ``incoming_references``."""
    return TableReference(
        table=owner_table,
        column=owner_column,
        ref_table=this_table,
        ref_column=this_column,
    )


class TestBuildDiagram:
    async def test_raises_when_path_does_not_resolve_to_a_table(self) -> None:
        describe = _describe_from({})
        with pytest.raises(DiagramError):
            await build_diagram(["users"], describe)

    async def test_single_table_renders_a_box_with_its_name(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)
        assert "users" in result.diagram
        assert "id" in result.diagram

    async def test_type_length_is_dropped_from_the_box(self) -> None:
        desc = _entity(
            "users",
            fields=[_field("name", type_="VARCHAR2(50 CHAR, 200 BYTE)", pk=True)],
        )
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)
        assert "VARCHAR2" in result.diagram
        assert "BYTE" not in result.diagram
        assert "(" not in result.diagram

    async def test_schema_qualified_table_shows_schema_dot_table(self) -> None:
        desc = _entity("users", schema="dbo")
        describe = _describe_from({("dbo", "users"): desc})
        result = await build_diagram(["dbo", "users"], describe)
        assert "dbo.users" in result.diagram

    async def test_pk_column_is_marked(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)
        assert "PK" in result.diagram

    async def test_fk_column_is_marked(self) -> None:
        desc = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                )
            ],
        )
        users = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("orders",): desc, ("users",): users})
        result = await build_diagram(["orders"], describe)
        assert "FK" in result.diagram

    async def test_many_to_one_edge_shows_star_and_one_markers(self) -> None:
        orders = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                )
            ],
        )
        users = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("orders",): orders, ("users",): users})
        result = await build_diagram(["orders"], describe)
        # "*" (many) at the FK side, "1" at the referenced side.
        assert re.search(r"│\*─*1│", result.diagram)

    async def test_one_to_one_edge_shows_one_markers_on_both_ends(self) -> None:
        orders = _entity(
            "orders",
            fields=[
                _field(
                    "user_id",
                    outgoing=[
                        _outgoing("orders", "user_id", "users", "id", unique=True)
                    ],
                )
            ],
        )
        users = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("orders",): orders, ("users",): users})
        result = await build_diagram(["orders"], describe)
        # FK column is itself unique, so both ends show "1" instead of "*".
        assert re.search(r"│1─*1│", result.diagram)

    async def test_outgoing_reference_renders_connected_table(self) -> None:
        orders = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                )
            ],
        )
        users = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("orders",): orders, ("users",): users})
        result = await build_diagram(["orders"], describe)
        assert "users" in result.diagram

    async def test_incoming_reference_renders_connected_table(self) -> None:
        users = _entity(
            "users",
            fields=[
                _field(
                    "id",
                    pk=True,
                    incoming=[_incoming("orders", "user_id", "users", "id")],
                )
            ],
        )
        orders = _entity("orders", fields=[_field("user_id")])
        describe = _describe_from({("users",): users, ("orders",): orders})
        result = await build_diagram(["users"], describe)
        assert "orders" in result.diagram

    async def test_self_reference_is_not_duplicated(self) -> None:
        desc = _entity(
            "employees",
            fields=[
                _field(
                    "manager_id",
                    outgoing=[_outgoing("employees", "manager_id", "employees", "id")],
                )
            ],
        )
        describe = _describe_from({("employees",): desc})
        result = await build_diagram(["employees"], describe)
        assert result.diagram.count("┌─ employees") == 1

    async def test_cycle_does_not_recurse_infinitely(self) -> None:
        a = _entity(
            "a",
            fields=[_field("b_id", outgoing=[_outgoing("a", "b_id", "b", "id")])],
        )
        b = _entity(
            "b",
            fields=[_field("a_id", outgoing=[_outgoing("b", "a_id", "a", "id")])],
        )
        describe = _describe_from({("a",): a, ("b",): b})
        result = await build_diagram(["a"], describe)
        assert result.diagram.count("┌─ a") == 1
        assert result.diagram.count("┌─ b") == 1

    async def test_unresolvable_reference_gets_a_placeholder_box(self) -> None:
        desc = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                )
            ],
        )
        describe = _describe_from({("orders",): desc})
        result = await build_diagram(["orders"], describe)
        assert "┌─ users" in result.diagram
        assert "(unavailable)" in result.diagram

    async def test_non_table_describe_result_raises(self) -> None:
        describe = _describe_from({("i",): IndexDescription(name="i", fields=[])})
        with pytest.raises(DiagramError):
            await build_diagram(["i"], describe)

    async def test_non_key_columns_are_collapsed_to_an_ellipsis(self) -> None:
        desc = _entity(
            "users",
            fields=[
                _field("id", pk=True),
                _field("name", "TEXT"),
                _field("email", "TEXT"),
            ],
        )
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)
        assert "..." in result.diagram
        assert "name" not in result.diagram
        assert "email" not in result.diagram

    async def test_incoming_fk_column_is_kept(self) -> None:
        users = _entity(
            "users",
            fields=[
                _field("id", pk=True),
                _field(
                    "username",
                    "TEXT",
                    incoming=[_incoming("posts", "author", "users", "username")],
                ),
                _field("bio", "TEXT"),
            ],
        )
        posts = _entity("posts", fields=[_field("author", "TEXT")])
        describe = _describe_from({("users",): users, ("posts",): posts})
        result = await build_diagram(["users"], describe)
        assert "username" in result.diagram
        assert "bio" not in result.diagram
        assert "..." in result.diagram

    async def test_no_ellipsis_when_all_columns_are_keys(self) -> None:
        desc = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                ),
                _field(
                    "product_id",
                    outgoing=[_outgoing("orders", "product_id", "products", "id")],
                ),
            ],
        )
        describe = _describe_from({("orders",): desc})
        result = await build_diagram(["orders"], describe)
        assert "..." not in result.diagram

    async def test_diamond_reference_draws_exactly_one_box(self) -> None:
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("a_id", outgoing=[_outgoing("root", "a_id", "a", "id")]),
                _field("b_id", outgoing=[_outgoing("root", "b_id", "b", "id")]),
            ],
        )
        a = _entity(
            "a",
            fields=[
                _field("id", pk=True),
                _field(
                    "shared_id",
                    outgoing=[_outgoing("a", "shared_id", "shared", "id")],
                ),
            ],
        )
        b = _entity(
            "b",
            fields=[
                _field("id", pk=True),
                _field(
                    "shared_id",
                    outgoing=[_outgoing("b", "shared_id", "shared", "id")],
                ),
            ],
        )
        shared = _entity("shared", fields=[_field("id", pk=True)])
        describe = _describe_from(
            {("root",): root, ("a",): a, ("b",): b, ("shared",): shared}
        )
        result = await build_diagram(["root"], describe)
        assert result.diagram.count("┌─ shared") == 1

    async def test_edge_between_boxes_of_different_heights_is_straight(self) -> None:
        tall = _entity(
            "tall",
            fields=[_field(f"k{i}", pk=True) for i in range(4)]
            + [
                _field(
                    "small_id", outgoing=[_outgoing("tall", "small_id", "small", "id")]
                )
            ],
        )
        small = _entity("small", fields=[_field("id", pk=True)])
        describe = _describe_from({("tall",): tall, ("small",): small})
        result = await build_diagram(["tall"], describe)
        # small is nudged down so both anchors share a row — one straight
        # border-to-border line, no jog in the channel. Endpoints carry
        # cardinality markers: "*" (many) at the FK side, "1" at the
        # referenced side.
        assert re.search(r"│\*─*1│", result.diagram)

    async def test_skip_edge_routes_past_the_intermediate_box_via_a_bypass_lane(
        self,
    ) -> None:
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("mid_id", outgoing=[_outgoing("root", "mid_id", "mid", "id")]),
                _field(
                    "leaf_id", outgoing=[_outgoing("root", "leaf_id", "leaf", "id")]
                ),
            ],
        )
        mid = _entity(
            "mid",
            fields=[
                _field("id", pk=True),
                _field("leaf_id", outgoing=[_outgoing("mid", "leaf_id", "leaf", "id")]),
            ],
        )
        leaf = _entity("leaf", fields=[_field("id", pk=True)])
        describe = _describe_from({("root",): root, ("mid",): mid, ("leaf",): leaf})
        result = await build_diagram(["root"], describe)
        # The root→leaf edge crosses mid's rank; it should bypass mid via a
        # side lane instead of forcing mid out of its natural row.
        lines = result.diagram.splitlines()
        row_of = {
            t: next(r for r, s in enumerate(lines) if f"┌─ {t} " in s)
            for t in ("root", "mid")
        }
        assert row_of["mid"] == row_of["root"]

    async def test_multiple_relationships_on_the_same_side_get_distinct_anchors(
        self,
    ) -> None:
        # a and c both land on the hub's same side (b takes the other side by
        # LPT balancing) — their edges must not both bunch at one anchor.
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("a_id", outgoing=[_outgoing("root", "a_id", "a", "id")]),
                _field("b_id", outgoing=[_outgoing("root", "b_id", "b", "id")]),
                _field("c_id", outgoing=[_outgoing("root", "c_id", "c", "id")]),
            ],
        )
        a = _entity("a", fields=[_field("id", pk=True)])
        b = _entity("b", fields=[_field("id", pk=True)])
        c = _entity("c", fields=[_field("id", pk=True)])
        describe = _describe_from({("root",): root, ("a",): a, ("b",): b, ("c",): c})
        result = await build_diagram(["root"], describe)

        rows_by_edge: dict[tuple[str, ...], set[int]] = {}
        for r in result.regions:
            if r.kind == "edge":
                rows_by_edge.setdefault(tuple(r.path), set()).add(r.row)
        a_rows = rows_by_edge[("root", "relationships", "a_id")]
        c_rows = rows_by_edge[("root", "relationships", "c_id")]
        assert min(a_rows) != min(c_rows)


class TestBuildDiagramRegions:
    async def test_table_header_region_resolves_to_table_path(self) -> None:
        desc = _entity("users", schema="dbo")
        describe = _describe_from({("dbo", "users"): desc})
        result = await build_diagram(["dbo", "users"], describe)

        region = next(r for r in result.regions if r.path == ["dbo", "users"])
        line = result.diagram.splitlines()[region.row]
        span = line.encode()[region.col_start : region.col_end].decode()
        assert "dbo.users" in span

    async def test_column_region_resolves_to_column_path(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        region = next(r for r in result.regions if r.path == ["users", "columns", "id"])
        line = result.diagram.splitlines()[region.row]
        span = line.encode()[region.col_start : region.col_end].decode()
        assert span == "id"

    async def test_column_region_offset_accounts_for_multibyte_box_chars(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        region = next(r for r in result.regions if r.path == ["users", "columns", "id"])
        line = result.diagram.splitlines()[region.row]
        assert line.startswith("│ id")
        # "│ " is 4 bytes (│ is 3-byte UTF-8) but only 2 characters — the
        # byte offset must reflect that, not the character index.
        assert region.col_start == 4

    async def test_table_referenced_twice_still_gets_drawn_as_one_box(self) -> None:
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("a_id", outgoing=[_outgoing("root", "a_id", "a", "id")]),
                _field("b_id", outgoing=[_outgoing("root", "b_id", "b", "id")]),
            ],
        )
        a = _entity(
            "a",
            fields=[
                _field("id", pk=True),
                _field(
                    "shared_id",
                    outgoing=[_outgoing("a", "shared_id", "shared", "id")],
                ),
            ],
        )
        b = _entity(
            "b",
            fields=[
                _field("id", pk=True),
                _field(
                    "shared_id",
                    outgoing=[_outgoing("b", "shared_id", "shared", "id")],
                ),
            ],
        )
        shared = _entity("shared", fields=[_field("id", pk=True)])
        describe = _describe_from(
            {("root",): root, ("a",): a, ("b",): b, ("shared",): shared}
        )
        result = await build_diagram(["root"], describe)

        # "shared" has one column, so a single box is 3 rows (top border,
        # one column row, bottom border); a duplicate box would add more rows.
        matches = [r for r in result.regions if r.path == ["shared"]]
        assert len({r.row for r in matches}) == 3

    async def test_ellipsis_region_resolves_to_the_table_column_list(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True), _field("name", "TEXT")])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        region = next(r for r in result.regions if r.path == ["users", "columns"])
        line = result.diagram.splitlines()[region.row]
        span = line.encode()[region.col_start : region.col_end].decode()
        assert span == "..."
        # kind, not path shape, is what tells the ellipsis apart from a leaf
        # column -- a table may itself have a column named "columns".
        assert region.kind == "columns"

    async def test_a_column_named_columns_is_still_a_leaf_column_region(self) -> None:
        # A key column literally named "columns" is drawn, and a non-key one is
        # hidden behind the "..." row, so both regions coexist on paths that
        # differ only in length: ["users", "columns"] for the ellipsis and
        # ["users", "columns", "columns"] for the real column.
        desc = _entity(
            "users",
            fields=[
                _field("id", pk=True),
                _field("columns", "TEXT", pk=True),
                _field("name", "TEXT"),
            ],
        )
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        kinds = {tuple(r.path): r.kind for r in result.regions}
        assert kinds[("users", "columns")] == "columns"
        assert kinds[("users", "columns", "columns")] == "column"

    async def test_unresolved_reference_still_gets_a_region(self) -> None:
        desc = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                )
            ],
        )
        describe = _describe_from({("orders",): desc})
        result = await build_diagram(["orders"], describe)

        region = next(r for r in result.regions if r.path == ["users"])
        line = result.diagram.splitlines()[region.row]
        span = line.encode()[region.col_start : region.col_end].decode()
        assert "users" in span

    async def test_table_header_region_has_table_kind(self) -> None:
        desc = _entity("users")
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        region = next(r for r in result.regions if r.path == ["users"])
        assert region.kind == "table"

    async def test_column_region_has_column_kind(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        region = next(r for r in result.regions if r.path == ["users", "columns", "id"])
        assert region.kind == "column"

    async def test_relationship_gets_an_edge_kind_region(self) -> None:
        orders = _entity(
            "orders",
            fields=[
                _field(
                    "user_id", outgoing=[_outgoing("orders", "user_id", "users", "id")]
                )
            ],
        )
        users = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("orders",): orders, ("users",): users})
        result = await build_diagram(["orders"], describe)

        edge_regions = [r for r in result.regions if r.kind == "edge"]
        assert len(edge_regions) == 1
        region = edge_regions[0]
        assert region.path == ["orders", "relationships", "user_id"]
        line = result.diagram.splitlines()[region.row]
        span = line.encode()[region.col_start : region.col_end].decode()
        assert re.fullmatch(r"\*─+1", span)

    async def test_multi_row_edge_regions_share_the_same_path(self) -> None:
        # The mid->leaf edge in the skip-edge fixture detours vertically past
        # root's rank, spanning several rows — every row it touches gets its
        # own region, all sharing the same path.
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("mid_id", outgoing=[_outgoing("root", "mid_id", "mid", "id")]),
                _field(
                    "leaf_id", outgoing=[_outgoing("root", "leaf_id", "leaf", "id")]
                ),
            ],
        )
        mid = _entity(
            "mid",
            fields=[
                _field("id", pk=True),
                _field("leaf_id", outgoing=[_outgoing("mid", "leaf_id", "leaf", "id")]),
            ],
        )
        leaf = _entity("leaf", fields=[_field("id", pk=True)])
        describe = _describe_from({("root",): root, ("mid",): mid, ("leaf",): leaf})
        result = await build_diagram(["root"], describe)

        edge_regions = [
            r
            for r in result.regions
            if r.kind == "edge" and r.path == ["mid", "relationships", "leaf_id"]
        ]
        assert len(edge_regions) > 1
        assert len({tuple(r.path) for r in edge_regions}) == 1

    async def test_table_box_border_gets_a_region_on_every_row(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        table_regions = [r for r in result.regions if r.path == ["users"]]
        # top border, one column row (left + right border), bottom border.
        assert len({r.row for r in table_regions}) == 3

    async def test_table_border_region_on_column_row_does_not_overlap_column(
        self,
    ) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        column_region = next(
            r for r in result.regions if r.path == ["users", "columns", "id"]
        )
        border_regions = [
            r
            for r in result.regions
            if r.path == ["users"] and r.row == column_region.row
        ]
        for border in border_regions:
            overlap = border.col_start < column_region.col_end and (
                column_region.col_start < border.col_end
            )
            assert not overlap

    async def test_table_regions_all_share_the_same_path(self) -> None:
        desc = _entity("users", fields=[_field("id", pk=True)])
        describe = _describe_from({("users",): desc})
        result = await build_diagram(["users"], describe)

        table_regions = [r for r in result.regions if r.kind == "table"]
        assert all(r.path == ["users"] for r in table_regions)

    async def test_every_interior_row_gets_a_left_and_right_border_region(self) -> None:
        desc = _entity("t", fields=[_field(n, pk=True) for n in "abcd"])
        describe = _describe_from({("t",): desc})
        result = await build_diagram(["t"], describe)

        table_regions = [r for r in result.regions if r.path == ["t"]]
        interior_rows = range(1, 5)  # rows between the top and bottom border
        for row in interior_rows:
            assert sum(1 for r in table_regions if r.row == row) == 2

    async def test_skip_edge_never_overlaps_an_unrelated_table_region(self) -> None:
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("mid_id", outgoing=[_outgoing("root", "mid_id", "mid", "id")]),
                _field(
                    "leaf_id", outgoing=[_outgoing("root", "leaf_id", "leaf", "id")]
                ),
            ],
        )
        mid = _entity(
            "mid",
            fields=[
                _field("id", pk=True),
                _field("leaf_id", outgoing=[_outgoing("mid", "leaf_id", "leaf", "id")]),
            ],
        )
        leaf = _entity("leaf", fields=[_field("id", pk=True)])
        describe = _describe_from({("root",): root, ("mid",): mid, ("leaf",): leaf})
        result = await build_diagram(["root"], describe)

        # The root->leaf edge skips over mid's rank; its path must never
        # share cells with mid's box, or the canvas would silently swallow
        # that stretch of the connector (box cells always win).
        skip_edge = [
            r
            for r in result.regions
            if r.kind == "edge" and r.path == ["root", "relationships", "leaf_id"]
        ]
        mid_regions = [r for r in result.regions if r.path == ["mid"]]
        for edge in skip_edge:
            for table in mid_regions:
                if edge.row != table.row:
                    continue
                overlap = edge.col_start < table.col_end and (
                    table.col_start < edge.col_end
                )
                assert not overlap

    async def test_edges_fanning_out_from_the_same_box_never_share_a_cell(
        self,
    ) -> None:
        root = _entity(
            "root",
            fields=[
                _field("id", pk=True),
                _field("a_id", outgoing=[_outgoing("root", "a_id", "a", "id")]),
                _field("b_id", outgoing=[_outgoing("root", "b_id", "b", "id")]),
                _field("c_id", outgoing=[_outgoing("root", "c_id", "c", "id")]),
            ],
        )
        a = _entity("a", fields=[_field("id", pk=True)])
        b = _entity(
            "b",
            fields=[
                _field("id", pk=True),
                _field("d_id", outgoing=[_outgoing("b", "d_id", "d", "id")]),
            ],
        )
        c = _entity("c", fields=[_field("id", pk=True)])
        d = _entity("d", fields=[_field("id", pk=True)])
        describe = _describe_from(
            {("root",): root, ("a",): a, ("b",): b, ("c",): c, ("d",): d}
        )
        result = await build_diagram(["root"], describe)

        # Every relationship leaving "root" shares its box side with the
        # others; none of their routed paths may occupy the same cell,
        # whichever direction each one has to jog to reach its target.
        cells_by_path: dict[tuple[str, ...], set[tuple[int, int]]] = {}
        for r in result.regions:
            if r.kind != "edge":
                continue
            cells = cells_by_path.setdefault(tuple(r.path), set())
            cells.update((r.row, c) for c in range(r.col_start, r.col_end))

        paths = list(cells_by_path)
        for i, path_a in enumerate(paths):
            for path_b in paths[i + 1 :]:
                assert not (cells_by_path[path_a] & cells_by_path[path_b])


class TestUnroutableEdges:
    """``route`` failing is not the end of the diagram: ``build_diagram``
    re-places with more room, and draws what it can if that still fails."""

    def _two_related_tables(self) -> Describe:
        parent = _entity("parent", fields=[_field("id", pk=True)])
        child = _entity(
            "child",
            fields=[
                _field("id", pk=True),
                _field(
                    "parent_id",
                    outgoing=[_outgoing("child", "parent_id", "parent", "id")],
                ),
            ],
        )
        return _describe_from({("child",): child, ("parent",): parent})

    async def test_retries_with_a_roomier_spacing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_place, real_route = diagram.place, diagram.route
        spacings: list[Spacing] = []
        calls = 0

        def spy_place(nodes, edges, layout, spacing) -> PlaceResult:  # type: ignore[no-untyped-def]
            spacings.append(spacing)
            return real_place(nodes, edges, layout, spacing)

        def flaky_route(nodes, edges, place_result):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if calls == 1:
                raise NoRouteError("no room on the first try")
            return real_route(nodes, edges, place_result)

        monkeypatch.setattr(diagram, "place", spy_place)
        monkeypatch.setattr(diagram, "route", flaky_route)
        result = await build_diagram(["child"], self._two_related_tables())

        assert len(spacings) == 2
        assert spacings[1].box_gap > spacings[0].box_gap
        assert "parent" in result.diagram

    async def test_fails_rather_than_drawing_a_diagram_missing_a_connector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = 0

        def stuck_route(nodes, edges, place_result):  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            raise NoRouteError("no room, ever")

        monkeypatch.setattr(diagram, "route", stuck_route)
        with pytest.raises(DiagramError):
            await build_diagram(["child"], self._two_related_tables())
        assert attempts == len(diagram._spacing_ladder(1))

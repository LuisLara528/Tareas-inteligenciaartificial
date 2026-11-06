"""
Ejercicio 1: Encadenamiento en la biblioteca
Implementacion autocontenida de Forward Chaining y Backward Chaining
Luis Lara Andrade
"""

import re

# 1. ESTRUCTURAS BÁSICAS (Átomos, Unificación y Estandarización)
class Atom:
    def __init__(self, pred, args):
        self.pred = pred
        self.args = tuple(args)

    def __repr__(self):
        return f"{self.pred}(" + ", ".join(self.args) + ")"

    def __eq__(self, other):
        return isinstance(other, Atom) and self.pred == other.pred and self.args == other.args

    def __hash__(self):
        return hash((self.pred, self.args))


def parse_atom(s: str) -> Atom:
    s = s.strip()
    m = re.match(r"([A-Za-z0-9_]+)\((.*)\)", s)
    if not m:
        raise ValueError(f"Átomo no válido: {s}")
    pred = m.group(1)
    args_str = m.group(2).strip()
    args = [a.strip() for a in args_str.split(",") if a.strip()] if args_str else []
    return Atom(pred, args)


def is_variable(x: str) -> bool:
    return len(x) > 0 and x[0].islower()


def unify(atom1: Atom, atom2: Atom, subst: dict | None = None) -> dict | None:
    if subst is None:
        subst = {}
    if atom1.pred != atom2.pred or len(atom1.args) != len(atom2.args):
        return None
    s = dict(subst)
    for a1, a2 in zip(atom1.args, atom2.args):
        val1 = s.get(a1, a1)
        val2 = s.get(a2, a2)
        if val1 == val2:
            continue
        elif is_variable(val1):
            s[val1] = val2
        elif is_variable(val2):
            s[val2] = val1
        else:
            return None
    return s


def apply_atom(atom: Atom, subst: dict) -> Atom:
    new_args = []
    for a in atom.args:
        cur = a
        while is_variable(cur) and cur in subst:
            cur = subst[cur]
        new_args.append(cur)
    return Atom(atom.pred, new_args)


class Rule:
    def __init__(self, rule_id: str, premises: list[Atom], head: Atom):
        self.rule_id = rule_id
        self.premises = tuple(premises)
        self.head = head

    def __repr__(self):
        return f"{self.rule_id}: " + " ^ ".join(str(p) for p in self.premises) + f" => {self.head}"


def standardize(rule: Rule, counter: int) -> Rule:
    mapping = {}

    def rename_atom(atom: Atom) -> Atom:
        new_args = []
        for arg in atom.args:
            if is_variable(arg):
                mapping.setdefault(arg, f"{arg}_{counter}")
                new_args.append(mapping[arg])
            else:
                new_args.append(arg)
        return Atom(atom.pred, new_args)

    return Rule(rule.rule_id, [rename_atom(p) for p in rule.premises], rename_atom(rule.head))


# 2. ENCADENAMIENTO HACIA ADELANTE (Forward Chaining)
def forward_chain(facts: list[Atom], rules: list[Rule], query: Atom):
    known = list(facts)
    inferred_order = []
    events = []
    counter = 0
    round_num = 1

    while True:
        new_this_round = []
        events.append(f"  --- Ronda {round_num} ---")
        start_facts = list(known)

        for rule in rules:
            def match_premises(premises, subst, f_list):
                if not premises:
                    yield subst
                    return
                first, rest = premises[0], premises[1:]
                for f in f_list:
                    th = unify(apply_atom(first, subst), f, subst)
                    if th is not None:
                        yield from match_premises(rest, th, f_list)

            counter += 1
            fresh = standardize(rule, counter)
            for theta in match_premises(fresh.premises, {}, start_facts):
                head_sub = apply_atom(fresh.head, theta)
                if head_sub not in known and head_sub not in new_this_round:
                    new_this_round.append(head_sub)
                    events.append(f"    Dispara {rule.rule_id}: deriva {head_sub} con {theta}")

        if not new_this_round:
            events.append("    (No se derivaron nuevos hechos en esta ronda)")
            break

        for a in new_this_round:
            known.append(a)
            inferred_order.append(a)
        round_num += 1

    proved = any(unify(query, f) is not None for f in known)
    return proved, inferred_order, events


# 3. ENCADENAMIENTO HACIA ATRÁS (Backward Chaining)
def backward_chain(facts: list[Atom], rules: list[Rule], query: Atom, max_depth: int = 25):
    events = []
    counter = [0]
    visited_goals = []

    def indent(depth):
        return "    " + "  " * depth

    def bc_or(goal: Atom, subst: dict, depth: int, stack: tuple):
        goal = apply_atom(goal, subst)
        key = str(goal)
        if key in stack:
            events.append(f"{indent(depth)}[Ciclo] {goal} ya esta en la pila. Se corta.")
            return None
        if depth > max_depth:
            events.append(f"{indent(depth)}[Profundidad maxima] en {goal}")
            return None

        events.append(f"{indent(depth)}Meta: {goal}")
        visited_goals.append(goal)

        # 1. Probar hechos primero
        for fact in facts:
            theta = unify(goal, fact, subst)
            if theta is not None:
                events.append(f"{indent(depth)}  -> Hecho confirmado: {fact}")
                return theta

        # 2. Probar reglas despues (en orden estricto)
        for rule in rules:
            counter[0] += 1
            fresh = standardize(rule, counter[0])
            theta = unify(goal, fresh.head, subst)
            if theta is None:
                continue
            events.append(f"{indent(depth)}  -> Prueba regla: {rule.rule_id} con cabeza {fresh.head}")
            answer = bc_and(fresh.premises, theta, depth + 1, stack + (key,))
            if answer is not None:
                return answer

        events.append(f"{indent(depth)}  [FALLO] No se pudo probar: {goal}")
        return None

    def bc_and(premises: tuple, subst: dict, depth: int, stack: tuple):
        if not premises:
            return subst
        first, rest = premises[0], premises[1:]
        theta = bc_or(first, subst, depth, stack)
        if theta is None:
            return None
        return bc_and(rest, theta, depth, stack)

    found = bc_or(query, {}, 0, ())
    proved = found is not None
    return proved, visited_goals, events


# 4. BASE DE CONOCIMIENTO (BIBLIOTECA) Y CONSULTAS
def get_library_kb():
    facts_raw = [
        "Estudiante(Ana)",
        "Estudiante(Luis)",
        "Profesor(Marta)",
        "TieneCredencial(Ana)",
        "TieneCredencial(Marta)",
        "HorarioNocturno(Hoy)",
    ]
    facts = [parse_atom(f) for f in facts_raw]

    rules_raw = [
        ("R_miembro_est", ["Estudiante(x)", "TieneCredencial(x)"], "Miembro(x)"),
        ("R_miembro_prof", ["Profesor(x)"], "Miembro(x)"),
        ("R_acceso", ["Miembro(x)", "HorarioNocturno(Hoy)"], "AccesoNoche(x)"),
        ("R_beca", ["Estudiante(x)", "TieneCredencial(x)"], "Becario(x)"),
    ]
    rules = [Rule(r[0], [parse_atom(p) for p in r[1]], parse_atom(r[2])) for r in rules_raw]
    return facts, rules


# 5. EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    facts, rules = get_library_kb()
    queries = [
        "AccesoNoche(Ana)",
        "AccesoNoche(Luis)",
        "AccesoNoche(Marta)",
    ]

    print("=" * 65)
    print("EJERCICIO 1: ENCADENAMIENTO EN LA BIBLIOTECA")
    print("=" * 65)

    for q_str in queries:
        query = parse_atom(q_str)
        print(f"\n")
        print(f"CONSULTA: {query}")
        print(f"\n")

        # Forward Chaining
        fc_proved, fc_inferred, fc_events = forward_chain(facts, rules, query)
        print("\n[A] ENCADENAMIENTO HACIA ADELANTE (Forward Chaining)")
        print(f"Resultado : {'PROBADA (SI)' if fc_proved else 'NO PROBADA (NO)'}")
        print("Atomos nuevos inferidos (en orden estricto):")
        for i, atom in enumerate(fc_inferred, 1):
            print(f"  {i}. {atom}")
        print("\nTraza paso a paso:")
        for ev in fc_events:
            print(ev)

        # Backward Chaining
        bc_proved, bc_visited, bc_events = backward_chain(facts, rules, query)
        print("\n[B] ENCADENAMIENTO HACIA ATRAS (Backward Chaining)")
        print(f"Resultado : {'PROBADA (SI)' if bc_proved else 'NO PROBADA (NO)'}")
        print("\nTraza paso a paso:")
        for ev in bc_events:
            print(ev)
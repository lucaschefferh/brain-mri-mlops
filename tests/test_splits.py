"""
Garantias sobre a divisão do dataset.

O projeto promete divisão por PACIENTE, não por fatia — fatias vizinhas do mesmo
exame são quase idênticas, então dividir por fatia inflaria as métricas de teste.
Até aqui essa promessa vivia num comentário do README; agora é executável.
"""

from itertools import combinations

from conftest import SPLITS, institution, patient_id

EXPECTED_PATIENTS = 110
EXPECTED_SLICES = 3929
EXPECTED_INSTITUTIONS = {"CS", "DU", "EZ", "FG", "HT"}


def test_nenhum_paciente_em_dois_splits(split_images):
    """A garantia central: um paciente pertence a exatamente um split."""
    patients = {
        split: {patient_id(p.name) for p in paths} for split, paths in split_images.items()
    }

    for a, b in combinations(SPLITS, 2):
        overlap = patients[a] & patients[b]
        assert not overlap, (
            f"Vazamento entre '{a}' e '{b}': {len(overlap)} paciente(s) em ambos. "
            f"Exemplos: {sorted(overlap)[:3]}"
        )


def test_todas_as_fatias_tem_mascara(split_images):
    """Cada imagem precisa do par `<stem>_mask.tif` no diretório irmão."""
    for split, paths in split_images.items():
        masks_dir = paths[0].parent.parent / "masks"
        faltando = [p.name for p in paths if not (masks_dir / f"{p.stem}_mask.tif").exists()]
        assert not faltando, (
            f"{len(faltando)} máscara(s) ausente(s) em '{split}'. Exemplos: {faltando[:3]}"
        )


def test_cobertura_do_dataset(split_images):
    """Os splits somados reconstroem o dataset original, sem perda nem duplicata."""
    todas = [p.name for paths in split_images.values() for p in paths]

    assert len(todas) == len(set(todas)), "há nomes de arquivo duplicados entre splits"
    assert len(todas) == EXPECTED_SLICES, f"esperava {EXPECTED_SLICES} fatias, achei {len(todas)}"

    pacientes = {patient_id(n) for n in todas}
    assert len(pacientes) == EXPECTED_PATIENTS, (
        f"esperava {EXPECTED_PATIENTS} pacientes, achei {len(pacientes)}"
    )


def test_instituicoes_conhecidas(split_images):
    """
    Trava a premissa da Fase 07: o holdout de drift é montado por instituição,
    então uma instituição inesperada precisa quebrar o teste, não passar batido.
    """
    encontradas = {
        institution(p.name) for paths in split_images.values() for p in paths
    }
    assert encontradas == EXPECTED_INSTITUTIONS, (
        f"instituições divergem do esperado: {sorted(encontradas)}"
    )

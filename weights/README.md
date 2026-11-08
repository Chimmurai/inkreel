# Ваги IRNM1

| Тека | Що | Де використовується |
|---|---|---|
| `ink1/best.pt` | основна модель fastq, 9 год навчання (EMA), вчитель Ultra | джерело для fine-tune і дистиляції (`--init-from ink1 --kd ink1`) |
| `lite_fq_b/best.pt` | легка модель fq_b, дотренована з ink1 (60 хв) | **стоїть у застосунку**: `benvid/assets/student` (sha f8a4b65c95f08d33) |
| `student_backup/` | ONNX попередньої моделі (ink1 fastq) | запасний варіант; `tools/install_student.py` кладе сюди старі збірки |

Повні прогони (чекпойнти, журнали, preview) лишаються поруч із датасетом: `E:\Inkreel_Dataset\runs\`.
У застосунок встановлює `InkReel.bat` → 3 → 5 (`tools/install_student.py ІМ'Я_ПРОГОНУ`).

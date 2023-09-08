The file `stack_overflow_postgresql.dump` is an official dump of the stackoverflow question data from MSR2013. 

To load the data, first create a database `db_name` to save the tables (or you can choose to reuse the existing database).

```bash
psql -U username -d db_name -f stack_overflow_postgresql.dump
```
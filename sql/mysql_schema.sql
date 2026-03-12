CREATE DATABASE IF NOT EXISTS `construction_db`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `construction_db`;

CREATE TABLE IF NOT EXISTS `zones` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(50) NOT NULL,
  `description` VARCHAR(100) NULL,
  `task` VARCHAR(100) NULL,
  `risk_level` VARCHAR(20) NULL DEFAULT 'safe',
  `max_workers` INT NULL DEFAULT 30,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `alerts` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `level` VARCHAR(20) NULL,
  `message` TEXT NULL,
  `source` VARCHAR(50) NULL,
  `is_resolved` BOOLEAN NULL DEFAULT FALSE,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `progress` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `task_name` VARCHAR(100) NULL,
  `percentage` INT NULL DEFAULT 0,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `reports` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `date` VARCHAR(20) NULL,
  `text_content` TEXT NULL,
  `translated_text` TEXT NULL,
  `source_language` VARCHAR(10) NULL DEFAULT 'ko',
  `target_language` VARCHAR(10) NULL,
  `author_name` VARCHAR(50) NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `translation_logs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `source_text` TEXT NOT NULL,
  `translated_text` TEXT NOT NULL,
  `source_language` VARCHAR(10) NOT NULL,
  `target_language` VARCHAR(10) NOT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_translation_logs_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `sensor_data` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `zone_id` INT NULL,
  `sensor_type` VARCHAR(50) NULL,
  `value` FLOAT NULL,
  `unit` VARCHAR(20) NULL,
  `recorded_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_sensor_data_zone_id` (`zone_id`),
  CONSTRAINT `fk_sensor_data_zone_id`
    FOREIGN KEY (`zone_id`) REFERENCES `zones` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `photos` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `zone_id` INT NULL,
  `file_path` VARCHAR(300) NULL,
  `original_name` VARCHAR(200) NULL,
  `ai_result` TEXT NULL,
  `risk_detected` BOOLEAN NULL DEFAULT FALSE,
  `taken_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_photos_zone_id` (`zone_id`),
  CONSTRAINT `fk_photos_zone_id`
    FOREIGN KEY (`zone_id`) REFERENCES `zones` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `workers` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(50) NOT NULL,
  `role` VARCHAR(50) NULL,
  `phone` VARCHAR(20) NULL,
  `zone_id` INT NULL,
  `status` VARCHAR(20) NULL DEFAULT 'work',
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_workers_zone_id` (`zone_id`),
  CONSTRAINT `fk_workers_zone_id`
    FOREIGN KEY (`zone_id`) REFERENCES `zones` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `zones` (`id`, `name`, `description`, `task`, `risk_level`, `max_workers`)
VALUES
  (1, 'Zone A', 'B2', 'Rebar Work', 'safe', 30),
  (2, 'Zone B', 'B1', 'Concrete', 'safe', 30),
  (3, 'Zone C', '1F-3F', 'High-altitude Work', 'caution', 30),
  (4, 'Zone D', '4F-6F', 'Frame Construction', 'safe', 30),
  (5, 'Zone E', 'Roof', 'Roof Work', 'danger', 30),
  (6, 'Zone F', 'Exterior', 'Facade Finishing', 'safe', 30)
ON DUPLICATE KEY UPDATE
  `name` = VALUES(`name`),
  `description` = VALUES(`description`),
  `task` = VALUES(`task`),
  `risk_level` = VALUES(`risk_level`),
  `max_workers` = VALUES(`max_workers`);


CREATE TABLE IF NOT EXISTS `users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(50) NOT NULL,
  `password_hash` VARCHAR(255) NOT NULL,
  `name` VARCHAR(50) NOT NULL,
  `role` VARCHAR(50) NOT NULL DEFAULT 'site_manager',
  `is_active` BOOLEAN NULL DEFAULT TRUE,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_users_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
